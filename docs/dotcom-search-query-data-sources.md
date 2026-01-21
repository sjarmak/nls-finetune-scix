# Dotcom Search Query Data Sources

This document describes the available data sources for collecting search queries from the Sourcegraph dotcom instance, useful for building training datasets for NLS query fine-tuning.

## Overview

Search queries from dotcom users are stored in BigQuery under the `telligentsourcegraph` project. The primary data is captured through analytics events that track user search behavior.

## Data Sources

### 1. `telligentsourcegraph.dotcom_events.events` (Primary Source)

The main events table containing real-time search activity.

**Key Characteristics:**
- Real-time event stream with timestamps
- Contains user context (anonymous user IDs, session info)
- Search query embedded in URL parameter
- Date range: June 2024 - present (ongoing ingestion)

**Relevant Event Names:**
| Event Name | Description | Volume |
|------------|-------------|--------|
| `SearchSubmitted` | User submitted a search query | ~174K events |
| `ViewSearchResults` | User viewed search results page | ~187K events |
| `SearchResultsQueried` | Search query was executed | ~320K events |
| `SearchResultsFetched` | Results were returned | ~396K events |
| `SearchResultClicked` | User clicked on a result | ~123K events |

**Schema (relevant columns):**
```
user_id          NUMERIC    - User identifier (if logged in)
name             STRING     - Event name
url              STRING     - Full URL containing the search query
anonymous_user_id STRING    - Anonymous tracking ID
timestamp        TIMESTAMP  - Event timestamp
public_argument  STRING     - JSON with metadata (source, patternType)
```

**Example Query:**
```sql
SELECT
  REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query,
  REGEXP_EXTRACT(url, r'patternType=([^&]+)') as pattern_type,
  JSON_EXTRACT_SCALAR(public_argument, '$.source') as search_source,
  timestamp
FROM `telligentsourcegraph.dotcom_events.events`
WHERE name = 'SearchSubmitted'
  AND url LIKE '%/search?%'
  AND url LIKE '%q=%'
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
LIMIT 1000
```

### 2. `telligentsourcegraph.dotcom_events.search_urls` (Historical Archive)

A dedicated table containing historical search URLs.

**Key Characteristics:**
- ~122K unique search URLs
- Simple schema (single `url` column)
- Good for historical diversity
- No timestamp information

**Schema:**
```
url    STRING    - Full search URL
```

**Example Query:**
```sql
SELECT
  REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query,
  REGEXP_EXTRACT(url, r'patternType=([^&]+)') as pattern_type
FROM `telligentsourcegraph.dotcom_events.search_urls`
WHERE url LIKE '%/search?%'
  AND url LIKE '%q=%'
LIMIT 1000
```

## Extracting Search Queries

### URL Structure

Search queries are embedded in the URL's `q` parameter:
```
https://sourcegraph.com/search?q=<encoded_query>&patternType=<type>&...
```

### URL Decoding

Queries are URL-encoded and need decoding:

| Encoded | Decoded |
|---------|---------|
| `%20` or `+` | space |
| `%3A` | `:` |
| `%2F` | `/` |
| `%5C` | `\` |
| `%5E` | `^` |
| `%24` | `$` |
| `%22` | `"` |
| `%27` | `'` |
| `%40` | `@` |
| `%7B` | `{` |
| `%7D` | `}` |
| `%5B` | `[` |
| `%5D` | `]` |

**BigQuery URL Decoding (partial):**
```sql
-- Basic decoding (for full decoding, use a UDF or post-process)
REPLACE(
  REPLACE(
    REPLACE(
      REPLACE(
        REPLACE(encoded_query, '%20', ' '),
      '%3A', ':'),
    '%2F', '/'),
  '%5C', '\\'),
'+', ' ') as decoded_query
```

### Comprehensive Extraction Query

```sql
-- Extract diverse search queries with metadata
WITH raw_queries AS (
  SELECT
    REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query,
    REGEXP_EXTRACT(url, r'patternType=([^&]+)') as pattern_type,
    JSON_EXTRACT_SCALAR(public_argument, '$.source') as source,
    timestamp,
    anonymous_user_id
  FROM `telligentsourcegraph.dotcom_events.events`
  WHERE name IN ('SearchSubmitted', 'ViewSearchResults')
    AND url LIKE '%/search?%'
    AND url LIKE '%q=%'
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
)
SELECT
  encoded_query,
  pattern_type,
  source,
  COUNT(*) as query_count,
  COUNT(DISTINCT anonymous_user_id) as unique_users,
  MIN(timestamp) as first_seen,
  MAX(timestamp) as last_seen
FROM raw_queries
WHERE encoded_query IS NOT NULL
  AND LENGTH(encoded_query) BETWEEN 5 AND 2000
GROUP BY encoded_query, pattern_type, source
ORDER BY query_count DESC
```

## Query Pattern Types

Sourcegraph supports multiple search pattern types:

| Pattern Type | Description | Example |
|--------------|-------------|---------|
| `keyword` | Keyword-based search (default for new users) | `error handling golang` |
| `literal` | Exact string matching | `"func main()"` |
| `regexp` | Regular expression patterns | `func\s+\w+\(` |
| `standard` | Traditional Sourcegraph search | `repo:github.com/... test` |
| `structural` | Structural code patterns | `'{ :[$_] assert(:[1]); }'` |

## Common Query Components

### Filters Found in Queries

- **Repository filters:** `repo:^github\.com/org/repo$`
- **Language filters:** `lang:TypeScript`, `lang:Go`, `lang:Python`
- **File filters:** `file:\.go$`, `file:^src/`
- **Context:** `context:global` (search across all repos)
- **Type filters:** `type:diff`, `type:commit`, `type:symbol`
- **Case sensitivity:** `case:yes`
- **Count limits:** `count:1000`, `count:all`
- **Time filters:** `repo:has.commit.after(last 12 months)`

### Query Sources (from `public_argument`)

- `nav` - Search from navigation bar
- `home` - Search from homepage
- `filter` - Search with filters applied
- `tree` - Search from file tree

## Data Quality Considerations

### Things to Filter Out

1. **Empty/minimal queries:** `context:global ` with no actual search term
2. **Bot/scanning traffic:** Queries with injection attempts or scanning patterns
3. **Test queries:** Queries from Sourcegraph teammates (if identifiable)
4. **Duplicate queries:** Same query from same user in quick succession

### Recommended Filters

```sql
WHERE
  -- Minimum query length
  LENGTH(encoded_query) > 10
  -- Exclude empty context-only queries
  AND encoded_query != 'context:global'
  AND encoded_query != 'context:global+'
  AND encoded_query != 'context:global '
  -- Exclude obvious injection attempts
  AND encoded_query NOT LIKE '%<script%'
  AND encoded_query NOT LIKE '%<esi:%'
```

## Privacy Considerations

- Queries may contain sensitive information (API keys, internal paths, etc.)
- Consider filtering queries containing common sensitive patterns
- Anonymous user IDs are pseudonymous but could potentially be correlated
- Repository names in queries may reveal private repository access

## Access Methods

### Using `bq` CLI

```bash
# Run a query
bq query --use_legacy_sql=false --format=prettyjson "
SELECT * FROM \`telligentsourcegraph.dotcom_events.events\`
WHERE name = 'SearchSubmitted'
LIMIT 10
"

# Export to file
bq query --use_legacy_sql=false --format=csv "..." > output.csv
```

### Using BigQuery Console

1. Go to https://console.cloud.google.com/bigquery
2. Select project `telligentsourcegraph`
3. Navigate to `dotcom_events` dataset
4. Run queries in the query editor

### Using bigquery-playground CLI

```bash
cd /Users/val/Desktop/sourcegraph-root/bigquery-playground
./explore query run <query_file.sql> --output results.json
```

## Sample Queries for Dataset Collection

### Get Recent Diverse Queries

```sql
SELECT
  encoded_query,
  pattern_type,
  COUNT(*) as occurrences
FROM (
  SELECT
    REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query,
    REGEXP_EXTRACT(url, r'patternType=([^&]+)') as pattern_type
  FROM `telligentsourcegraph.dotcom_events.events`
  WHERE name = 'SearchSubmitted'
    AND url LIKE '%/search?%'
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
)
WHERE encoded_query IS NOT NULL
  AND LENGTH(encoded_query) > 10
GROUP BY encoded_query, pattern_type
HAVING occurrences >= 2  -- Filter out one-off queries
ORDER BY RAND()
LIMIT 10000
```

### Get Queries by Pattern Type

```sql
SELECT
  REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as query,
  timestamp
FROM `telligentsourcegraph.dotcom_events.events`
WHERE name = 'SearchSubmitted'
  AND url LIKE '%patternType=keyword%'
  AND timestamp >= '2025-01-01'
ORDER BY RAND()
LIMIT 1000
```

### Get Natural Language-like Queries

```sql
-- Queries that don't start with filters (more likely to be NL)
SELECT encoded_query
FROM (
  SELECT REGEXP_EXTRACT(url, r'[?&]q=([^&]+)') as encoded_query
  FROM `telligentsourcegraph.dotcom_events.events`
  WHERE name = 'SearchSubmitted'
    AND url LIKE '%/search?%'
    AND timestamp >= '2025-01-01'
)
WHERE encoded_query IS NOT NULL
  AND encoded_query NOT LIKE 'context:%'
  AND encoded_query NOT LIKE 'repo:%'
  AND encoded_query NOT LIKE 'file:%'
  AND encoded_query NOT LIKE 'lang:%'
  AND LENGTH(encoded_query) > 5
ORDER BY RAND()
LIMIT 1000
```

## Related Tables

Other potentially useful tables in `telligentsourcegraph`:

| Dataset.Table | Description |
|---------------|-------------|
| `dotcom_events.amplitude_events_prod_v3` | Amplitude analytics (less query detail) |
| `dotcom_events.cody` | Cody-specific events |
| `amp.events` | AMP/Cody tab events (different schema) |

## References

- BigQuery Playground repo: `/Users/val/Desktop/sourcegraph-root/bigquery-playground`
- Existing investigations: `bigquery-playground/investigations/`
- Shared queries: `bigquery-playground/shared/queries/`
