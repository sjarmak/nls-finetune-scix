import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type DatasetExample } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/CopyButton'

type SourceFilter = 'all' | 'gold' | 'generated'

export function DatasetBrowser() {
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
  const [categoryFilter, setCategoryFilter] = useState<string>('')

  const { data: stats } = useQuery({
    queryKey: ['dataset-stats'],
    queryFn: api.getDatasetStats,
  })

  const { data: examples, isLoading } = useQuery({
    queryKey: ['dataset-examples', sourceFilter, categoryFilter],
    queryFn: () =>
      api.listExamples({
        split: sourceFilter,
        category: categoryFilter || undefined,
        limit: 100,
      }),
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Dataset Browser</h1>
        <p className="text-muted-foreground">View and manage training examples</p>
      </div>

      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Gold Examples</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.by_source?.gold || 0}</div>
              <p className="text-xs text-muted-foreground">Hand-curated seed data</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Generated Examples</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.by_source?.generated || 0}</div>
              <p className="text-xs text-muted-foreground">From NL pair generation</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Training Set</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.train_examples}</div>
              <p className="text-xs text-muted-foreground">90% split for training</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Validation Set</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.val_examples}</div>
              <p className="text-xs text-muted-foreground">10% split for evaluation</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Category breakdown */}
      {stats && Object.keys(stats.by_type).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">By Category</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {Object.entries(stats.by_type)
                .sort(([, a], [, b]) => (b as number) - (a as number))
                .map(([category, count]) => (
                  <button
                    key={category}
                    onClick={() => setCategoryFilter(categoryFilter === category ? '' : category)}
                    className={`px-2 py-1 text-xs rounded ${
                      categoryFilter === category
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                  >
                    {category}: {count as number}
                  </button>
                ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex gap-1">
          {(['all', 'gold', 'generated'] as const).map((filter) => (
            <Button
              key={filter}
              variant={sourceFilter === filter ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSourceFilter(filter)}
            >
              {filter === 'all' ? 'All' : filter === 'gold' ? 'Gold' : 'Generated'}
              {stats && (
                <span className="ml-1 text-xs opacity-70">
                  ({filter === 'all'
                    ? stats.total_examples
                    : filter === 'gold'
                      ? stats.by_source?.gold || 0
                      : stats.by_source?.generated || 0})
                </span>
              )}
            </Button>
          ))}
        </div>
        {categoryFilter && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Category:</span>
            <span className="px-2 py-1 text-xs bg-primary text-primary-foreground rounded">
              {categoryFilter}
            </span>
            <button
              onClick={() => setCategoryFilter('')}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear
            </button>
          </div>
        )}
        <span className="text-sm text-muted-foreground ml-auto">
          Showing {examples?.length || 0} examples
        </span>
      </div>

      {/* Examples list */}
      <div className="space-y-2">
        {isLoading ? (
          <div className="text-center text-muted-foreground py-8">Loading...</div>
        ) : examples?.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">No examples found</div>
        ) : (
          examples?.map((example: DatasetExample) => (
            <ExampleCard key={example.id} example={example} />
          ))
        )}
      </div>
    </div>
  )
}

function ExampleCard({ example }: { example: DatasetExample }) {
  return (
    <div className="border rounded text-sm">
      {/* Header: source badge left, category + ID right */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/30">
        <span
          className={`px-1.5 py-0.5 text-xs rounded ${
            example.source === 'gold'
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-blue-100 text-blue-800'
          }`}
        >
          {example.source}
        </span>
        <div className="flex items-center gap-2">
          {example.category && (
            <span className="px-1.5 py-0.5 text-xs bg-muted rounded">{example.category}</span>
          )}
          <span className="text-xs text-muted-foreground">{example.id}</span>
        </div>
      </div>

      {/* Content rows - all same size, simple layout */}
      <div className="px-3 py-2 space-y-1 text-xs">
        <div className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-muted-foreground">Query</span>
          <span className="flex-1 min-w-0 truncate">{example.user_query}</span>
          <CopyButton text={example.user_query} />
        </div>
        <div className="flex items-center gap-2">
          <span className="w-20 shrink-0 text-muted-foreground">Expected</span>
          <code className="flex-1 min-w-0 font-mono truncate">{example.expected_output.query}</code>
          <CopyButton text={example.expected_output.query} />
        </div>
        {example.candidates.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="w-20 shrink-0 text-muted-foreground">Candidates</span>
            <span className="flex-1 min-w-0 text-muted-foreground truncate">{example.candidates.join(', ')}</span>
          </div>
        )}
      </div>
    </div>
  )
}
