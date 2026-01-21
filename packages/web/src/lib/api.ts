const API_BASE = '/api'

export interface InferenceRequest {
  query: string
  candidates: string[]
  date?: string
  model_id: string
}

export interface InferenceResponse {
  sourcegraph_query: string
  model_id: string
  latency_ms: number
}

export interface CompareRequest {
  query: string
  candidates: string[]
  date?: string
  model_ids: string[]
}

export interface CompareResponse {
  results: InferenceResponse[]
}

export interface ModelConfig {
  id: string
  name: string
  provider: string
  description: string
  is_available: boolean
}

export interface DatasetExample {
  id: string
  user_query: string
  candidates: string[]
  date: string
  expected_output: { query: string }
  created_at?: string
  source: string
  category?: string
}

export interface DatasetStats {
  total_examples: number
  train_examples: number
  val_examples: number
  by_type: Record<string, number>
  by_source: Record<string, number>
}

// New CLI-based evaluation types
export interface ModelMetrics {
  syntax_valid: number
  semantic_match: number
  avg_latency_ms: number
  total?: number
}

export interface EvalSummary {
  total: number
  fine_tuned: ModelMetrics
  baseline?: ModelMetrics
}

export interface ModelOutput {
  output: string
  syntax_valid: boolean
  hallucinations: string[]
  semantic_match: boolean
  latency_ms: number
}

export interface EvaluationResult {
  id: string
  input: string
  expected: string
  fine_tuned: ModelOutput
  baseline?: ModelOutput
  verdict: 'fine_tuned_better' | 'baseline_better' | 'tie' | 'both_wrong'
}

export interface EvaluationRun {
  id: string
  timestamp: string
  models: { fine_tuned: string; baseline?: string }
  summary: EvalSummary
  results: EvaluationResult[]
}

export interface EvaluationRunSummary {
  id: string
  timestamp: string
  model: string
  total: number
  syntax_valid_pct: number
  semantic_match_pct: number
  avg_latency_ms: number
  status: 'pass' | 'fail' | 'needs_improvement'
}

export const api = {
  // Inference
  async generate(request: InferenceRequest): Promise<InferenceResponse> {
    const res = await fetch(`${API_BASE}/inference/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return res.json()
  },

  async compare(request: CompareRequest): Promise<CompareResponse> {
    const res = await fetch(`${API_BASE}/inference/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return res.json()
  },

  // Models
  async listModels(): Promise<ModelConfig[]> {
    const res = await fetch(`${API_BASE}/models`)
    return res.json()
  },

  // Datasets
  async getDatasetStats(): Promise<DatasetStats> {
    const res = await fetch(`${API_BASE}/datasets/stats`)
    return res.json()
  },

  async listExamples(params?: {
    split?: string
    source?: string
    category?: string
    limit?: number
    offset?: number
  }): Promise<DatasetExample[]> {
    const searchParams = new URLSearchParams()
    if (params?.split) searchParams.set('split', params.split)
    if (params?.source) searchParams.set('source', params.source)
    if (params?.category) searchParams.set('category', params.category)
    if (params?.limit) searchParams.set('limit', params.limit.toString())
    if (params?.offset) searchParams.set('offset', params.offset.toString())

    const res = await fetch(`${API_BASE}/datasets/examples?${searchParams}`)
    return res.json()
  },

  async createExample(example: Omit<DatasetExample, 'id'>): Promise<DatasetExample> {
    const res = await fetch(`${API_BASE}/datasets/examples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(example),
    })
    return res.json()
  },

  async deleteExample(id: string): Promise<void> {
    await fetch(`${API_BASE}/datasets/examples/${id}`, { method: 'DELETE' })
  },

  // Evaluation (read-only - runs via CLI)
  async listEvaluationRuns(limit = 10): Promise<EvaluationRunSummary[]> {
    const res = await fetch(`${API_BASE}/evaluation/runs?limit=${limit}`)
    return res.json()
  },

  async getEvaluationRun(runId: string): Promise<EvaluationRun> {
    const res = await fetch(`${API_BASE}/evaluation/runs/${runId}`)
    return res.json()
  },

  async getLatestEvaluationRun(): Promise<EvaluationRun> {
    const res = await fetch(`${API_BASE}/evaluation/runs/latest`)
    return res.json()
  },
}
