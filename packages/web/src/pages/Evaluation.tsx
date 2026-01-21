import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  type EvaluationRunSummary,
  type EvaluationRun,
  type EvaluationResult,
} from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CopyButton } from '@/components/CopyButton'

// Success criteria targets
const TARGETS = {
  syntax: 95,
  semantic: 70,
  latency: 100,
}

type FilterType = 'all' | 'syntax_issues' | 'semantic_failures' | 'both_wrong'

export function Evaluation() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterType>('all')

  // Fetch list of runs
  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['evaluation-runs'],
    queryFn: () => api.listEvaluationRuns(10),
  })

  // Fetch selected run details
  const { data: selectedRun } = useQuery({
    queryKey: ['evaluation-run', selectedRunId],
    queryFn: () => (selectedRunId ? api.getEvaluationRun(selectedRunId) : null),
    enabled: !!selectedRunId,
  })

  // Filter results based on selected filter
  const filteredResults = selectedRun?.results.filter((r) => {
    switch (filter) {
      case 'syntax_issues':
        return !r.fine_tuned.syntax_valid
      case 'semantic_failures':
        return !r.fine_tuned.semantic_match
      case 'both_wrong':
        return r.verdict === 'both_wrong'
      default:
        return true
    }
  })

  if (runsLoading) {
    return <div className="p-8 text-center text-muted-foreground">Loading...</div>
  }

  // Show run detail if selected
  if (selectedRunId && selectedRun) {
    return (
      <RunDetail
        run={selectedRun}
        filter={filter}
        setFilter={setFilter}
        filteredResults={filteredResults || []}
        onBack={() => {
          setSelectedRunId(null)
          setFilter('all')
        }}
      />
    )
  }

  // Show runs list
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Evaluation</h1>
        <p className="text-muted-foreground">
          View model evaluation results. Run evaluations via CLI: <code>nls-finetune eval run</code>
        </p>
      </div>

      {!runs?.length ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No evaluation runs found. Run <code>nls-finetune eval run</code> to generate results.
          </CardContent>
        </Card>
      ) : (
        <RunsList runs={runs} onSelectRun={setSelectedRunId} />
      )}
    </div>
  )
}

// web-eval-001: Runs list with columns
function RunsList({
  runs,
  onSelectRun,
}: {
  runs: EvaluationRunSummary[]
  onSelectRun: (id: string) => void
}) {
  return (
    <div className="border rounded-lg overflow-hidden">
      <table className="w-full">
        <thead className="bg-muted/50">
          <tr>
            <th className="px-4 py-3 text-left text-sm font-medium">Run ID</th>
            <th className="px-4 py-3 text-left text-sm font-medium">Date</th>
            <th className="px-4 py-3 text-left text-sm font-medium">Model</th>
            <th className="px-4 py-3 text-right text-sm font-medium">Syntax Valid %</th>
            <th className="px-4 py-3 text-right text-sm font-medium">Semantic Match %</th>
            <th className="px-4 py-3 text-right text-sm font-medium">Avg Latency</th>
            <th className="px-4 py-3 text-center text-sm font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              className="border-t hover:bg-muted/30 cursor-pointer"
              onClick={() => onSelectRun(run.id)}
            >
              <td className="px-4 py-3 font-mono text-sm">{run.id}</td>
              <td className="px-4 py-3 text-sm">
                {new Date(run.timestamp).toLocaleString()}
              </td>
              <td className="px-4 py-3 text-sm truncate max-w-[200px]" title={run.model}>
                {run.model.split('/').pop() || run.model}
              </td>
              <td className="px-4 py-3 text-right text-sm">
                <span className={run.syntax_valid_pct >= TARGETS.syntax ? 'text-green-600' : 'text-red-600'}>
                  {run.syntax_valid_pct.toFixed(1)}%
                </span>
              </td>
              <td className="px-4 py-3 text-right text-sm">
                <span className={run.semantic_match_pct >= TARGETS.semantic ? 'text-green-600' : 'text-red-600'}>
                  {run.semantic_match_pct.toFixed(1)}%
                </span>
              </td>
              <td className="px-4 py-3 text-right text-sm">
                <span className={run.avg_latency_ms <= TARGETS.latency ? 'text-green-600' : run.avg_latency_ms <= 500 ? 'text-yellow-600' : 'text-red-600'}>
                  {run.avg_latency_ms.toFixed(0)}ms
                </span>
              </td>
              <td className="px-4 py-3 text-center">
                <StatusBadge status={run.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// web-eval-002, web-eval-004, web-eval-005: Run detail with results table, filters, summary
function RunDetail({
  run,
  filter,
  setFilter,
  filteredResults,
  onBack,
}: {
  run: EvaluationRun
  filter: FilterType
  setFilter: (f: FilterType) => void
  filteredResults: EvaluationResult[]
  onBack: () => void
}) {
  const summary = run.summary
  const ft = summary.fine_tuned
  const total = summary.total

  const syntaxPct = (ft.syntax_valid / total) * 100
  const semanticPct = (ft.semantic_match / total) * 100

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        <div>
          <h1 className="text-2xl font-bold">{run.id}</h1>
          <p className="text-sm text-muted-foreground">
            {new Date(run.timestamp).toLocaleString()} - {total} examples
          </p>
        </div>
      </div>

      {/* web-eval-005: Pass/fail summary against targets */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Syntax Valid"
          value={`${syntaxPct.toFixed(1)}%`}
          target={`${TARGETS.syntax}%`}
          pass={syntaxPct >= TARGETS.syntax}
        />
        <MetricCard
          label="Semantic Match"
          value={`${semanticPct.toFixed(1)}%`}
          target={`${TARGETS.semantic}%`}
          pass={semanticPct >= TARGETS.semantic}
        />
        <MetricCard
          label="Avg Latency"
          value={`${ft.avg_latency_ms.toFixed(0)}ms`}
          target={`${TARGETS.latency}ms`}
          pass={ft.avg_latency_ms <= TARGETS.latency}
        />
      </div>

      {/* web-eval-004: Filter dropdown */}
      <div className="flex items-center gap-4">
        <label className="text-sm font-medium">Filter:</label>
        <select
          className="border rounded px-3 py-1.5 text-sm"
          value={filter}
          onChange={(e) => setFilter(e.target.value as FilterType)}
        >
          <option value="all">All ({run.results.length})</option>
          <option value="syntax_issues">
            Syntax Issues ({run.results.filter((r) => !r.fine_tuned.syntax_valid).length})
          </option>
          <option value="semantic_failures">
            Semantic Failures ({run.results.filter((r) => !r.fine_tuned.semantic_match).length})
          </option>
          <option value="both_wrong">
            Both Wrong ({run.results.filter((r) => r.verdict === 'both_wrong').length})
          </option>
        </select>
        <span className="text-sm text-muted-foreground">
          Showing {filteredResults.length} results
        </span>
      </div>

      {/* Results - compact card layout showing all important info */}
      <div className="space-y-3">
        {filteredResults.map((result) => (
          <CompactResultCard key={result.id} result={result} models={run.models} />
        ))}
      </div>

    </div>
  )
}

// Extract short display name from model identifier
function getModelDisplayName(modelId: string): string {
  // Handle Modal URLs - our fine-tuned Qwen3 model
  if (modelId.includes('modal.run')) {
    return 'Qwen3-1.7B'
  }
  // Handle OpenAI models like "gpt-4o-mini"
  if (modelId.startsWith('gpt-')) return modelId
  // Handle other formats - take last part after / or return as-is
  const lastPart = modelId.split('/').pop() || modelId
  return lastPart.length > 20 ? lastPart.slice(0, 17) + '...' : lastPart
}

// Compact result card - 4 rows: Query, Expected, Model1, Model2
function CompactResultCard({
  result,
  models,
}: {
  result: EvaluationResult
  models: { fine_tuned: string; baseline?: string }
}) {
  const fineTunedName = getModelDisplayName(models.fine_tuned)
  const baselineName = models.baseline ? getModelDisplayName(models.baseline) : 'Baseline'

  return (
    <div className="border rounded overflow-hidden text-xs">
      {/* Row 1: Query (input) */}
      <div className="flex items-center gap-2 px-2 py-1 bg-muted/40 border-b">
        <span className="text-muted-foreground w-20 shrink-0 font-medium">Query</span>
        <span className="flex-1 truncate font-medium" title={result.input}>{result.input}</span>
        <VerdictBadge verdict={result.verdict} />
      </div>

      {/* Row 2: Expected */}
      <div className="flex items-center gap-2 px-2 py-1 border-b bg-muted/20">
        <span className="text-muted-foreground w-20 shrink-0">Expected</span>
        <code className="flex-1 font-mono truncate text-[11px]" title={result.expected}>{result.expected}</code>
        <CopyButton text={result.expected} />
      </div>

      {/* Row 3: Fine-tuned model */}
      <div className={`flex items-center gap-2 px-2 py-1 border-b ${
        result.fine_tuned.semantic_match ? 'bg-green-50' : 'bg-red-50'
      }`}>
        <span className="text-muted-foreground w-20 shrink-0 truncate" title={models.fine_tuned}>{fineTunedName}</span>
        <code className="flex-1 font-mono truncate text-[11px]" title={result.fine_tuned.output}>{result.fine_tuned.output}</code>
        <span className="text-muted-foreground shrink-0 w-14 text-right">{result.fine_tuned.latency_ms.toFixed(0)}ms</span>
        {result.fine_tuned.semantic_match ? (
          <span className="text-green-600 shrink-0">✓</span>
        ) : (
          <span className="text-red-600 shrink-0">✗</span>
        )}
        <CopyButton text={result.fine_tuned.output} />
      </div>

      {/* Row 4: Baseline model */}
      {result.baseline ? (
        <div className={`flex items-center gap-2 px-2 py-1 ${
          result.baseline.semantic_match ? 'bg-green-50' : 'bg-red-50'
        }`}>
          <span className="text-muted-foreground w-20 shrink-0 truncate" title={models.baseline}>{baselineName}</span>
          <code className="flex-1 font-mono truncate text-[11px]" title={result.baseline.output}>{result.baseline.output}</code>
          <span className="text-muted-foreground shrink-0 w-14 text-right">{result.baseline.latency_ms.toFixed(0)}ms</span>
          {result.baseline.semantic_match ? (
            <span className="text-green-600 shrink-0">✓</span>
          ) : (
            <span className="text-red-600 shrink-0">✗</span>
          )}
          <CopyButton text={result.baseline.output} />
        </div>
      ) : (
        <div className="flex items-center gap-2 px-2 py-1 bg-muted/20">
          <span className="text-muted-foreground w-20 shrink-0">No baseline</span>
          <span className="text-muted-foreground italic">—</span>
        </div>
      )}
    </div>
  )
}

function MetricCard({
  label,
  value,
  target,
  pass,
}: {
  label: string
  value: string
  target: string
  pass: boolean
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <span className="text-2xl font-bold">{value}</span>
          <div className="text-right">
            <div className="text-xs text-muted-foreground">Target: {target}</div>
            <div className={`text-lg ${pass ? 'text-green-600' : 'text-red-600'}`}>
              {pass ? '✓' : '✗'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors = {
    pass: 'bg-green-100 text-green-800',
    fail: 'bg-red-100 text-red-800',
    needs_improvement: 'bg-yellow-100 text-yellow-800',
  }
  const labels = {
    pass: 'PASS',
    fail: 'FAIL',
    needs_improvement: 'NEEDS WORK',
  }
  const tooltips = {
    pass: 'All metrics meet targets: Syntax ≥95%, Semantic ≥70%, Latency ≤100ms',
    fail: 'One or more metrics significantly below target',
    needs_improvement: 'Close to targets but not fully passing',
  }
  return (
    <span
      className={`px-2 py-1 rounded text-xs font-medium cursor-help ${colors[status as keyof typeof colors] || 'bg-gray-100'}`}
      title={tooltips[status as keyof typeof tooltips] || status}
    >
      {labels[status as keyof typeof labels] || status}
    </span>
  )
}

function VerdictBadge({ verdict, large }: { verdict: string; large?: boolean }) {
  const colors = {
    fine_tuned_better: 'bg-green-100 text-green-800',
    baseline_better: 'bg-blue-100 text-blue-800',
    tie: 'bg-gray-100 text-gray-800',
    both_wrong: 'bg-red-100 text-red-800',
  }
  const labels = {
    fine_tuned_better: 'Fine-tuned Better',
    baseline_better: 'Baseline Better',
    tie: 'Tie',
    both_wrong: 'Both Wrong',
  }
  const tooltips = {
    fine_tuned_better: 'Fine-tuned model produced correct result, baseline did not',
    baseline_better: 'Baseline model produced correct result, fine-tuned did not',
    tie: 'Both models produced semantically equivalent results',
    both_wrong: 'Neither model produced a semantically correct query',
  }
  return (
    <span
      className={`px-2 py-1 rounded font-medium cursor-help ${colors[verdict as keyof typeof colors] || 'bg-gray-100'} ${large ? 'text-sm' : 'text-xs'}`}
      title={tooltips[verdict as keyof typeof tooltips] || verdict}
    >
      {labels[verdict as keyof typeof labels] || verdict}
    </span>
  )
}
