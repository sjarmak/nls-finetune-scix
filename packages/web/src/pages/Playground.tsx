import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { api, type InferenceResponse, type ModelConfig } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CopyButton } from '@/components/CopyButton'

// Example queries for new users to try
const EXAMPLES = [
  { query: 'find authentication middleware in django', candidates: 'github.com/django/django' },
  { query: 'python test files in sourcegraph', candidates: 'github.com/sourcegraph/sourcegraph' },
  { query: 'recent commits by author erik', candidates: 'github.com/sourcegraph/sourcegraph' },
]

export function Playground() {
  const [query, setQuery] = useState('')
  const [candidates, setCandidates] = useState('')
  const [selectedModels, setSelectedModels] = useState<string[]>(['fine-tuned', 'gpt-4o-mini'])

  const loadExample = (example: typeof EXAMPLES[0]) => {
    setQuery(example.query)
    setCandidates(example.candidates)
  }

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: api.listModels,
  })

  const compareMutation = useMutation({
    mutationFn: () =>
      api.compare({
        query,
        candidates: candidates.split(',').map((c) => c.trim()).filter(Boolean),
        model_ids: selectedModels,
      }),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    compareMutation.mutate()
  }

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId]
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Playground</h1>
        <p className="text-muted-foreground">
          Test natural language to Sourcegraph query conversion
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Natural Language Query</label>
          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., find authentication code in sourcegraph"
            rows={3}
          />
          {!query.trim() && (
            <div className="mt-2">
              <span className="text-xs text-muted-foreground">Try an example: </span>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => loadExample(ex)}
                  className="text-xs text-blue-600 hover:underline mr-3"
                >
                  {ex.query}
                </button>
              ))}
            </div>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Repository Candidates
            <span className="text-muted-foreground font-normal ml-1">(optional)</span>
          </label>
          <Input
            value={candidates}
            onChange={(e) => setCandidates(e.target.value)}
            placeholder="e.g., github.com/sourcegraph/sourcegraph"
          />
          <p className="text-xs text-muted-foreground mt-1">
            Comma-separated repo URLs to scope the query. Leave empty to search all repositories.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Models to Compare</label>
          <div className="flex flex-wrap gap-2">
            {models?.map((model: ModelConfig) => (
              <Button
                key={model.id}
                type="button"
                variant={selectedModels.includes(model.id) ? 'default' : 'outline'}
                size="sm"
                onClick={() => toggleModel(model.id)}
              >
                {model.name}
              </Button>
            ))}
          </div>
        </div>

        <Button
          type="submit"
          disabled={!query.trim() || compareMutation.isPending}
          title={!query.trim() ? 'Enter a natural language query first' : undefined}
        >
          {compareMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Queries'
          )}
        </Button>
      </form>

      {compareMutation.data && (
        <div className="grid gap-4 md:grid-cols-2">
          {compareMutation.data.results.map((result: InferenceResponse) => (
            <Card key={result.model_id}>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center justify-between">
                  {models?.find((m: ModelConfig) => m.id === result.model_id)?.name || result.model_id}
                  <span className="text-sm font-normal text-muted-foreground">
                    {result.latency_ms.toFixed(0)}ms
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <pre className="bg-muted p-3 pr-10 rounded-md text-sm overflow-x-auto">
                    {result.sourcegraph_query}
                  </pre>
                  <div className="absolute top-2 right-2">
                    <CopyButton text={result.sourcegraph_query} />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
