import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { Playground } from '@/pages/Playground'
import { DatasetBrowser } from '@/pages/DatasetBrowser'
import { Evaluation } from '@/pages/Evaluation'
import { ComponentPlayground } from '@/pages/ComponentPlayground'

const queryClient = new QueryClient()

type Page = 'playground' | 'dataset' | 'evaluation' | 'components'

function App() {
  const [page, setPage] = useState<Page>('playground')

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background">
        <nav className="border-b">
          <div className="container mx-auto px-4">
            <div className="flex h-16 items-center gap-8">
              <span className="font-bold">NLS Query Fine-tune</span>
              <div className="flex gap-4">
                <button
                  onClick={() => setPage('playground')}
                  className={`text-sm ${page === 'playground' ? 'font-medium' : 'text-muted-foreground'}`}
                >
                  Playground
                </button>
                <button
                  onClick={() => setPage('dataset')}
                  className={`text-sm ${page === 'dataset' ? 'font-medium' : 'text-muted-foreground'}`}
                >
                  Dataset
                </button>
                <button
                  onClick={() => setPage('evaluation')}
                  className={`text-sm ${page === 'evaluation' ? 'font-medium' : 'text-muted-foreground'}`}
                >
                  Evaluation
                </button>
                <button
                  onClick={() => setPage('components')}
                  className={`text-sm ${page === 'components' ? 'font-medium' : 'text-muted-foreground'}`}
                >
                  Components
                </button>
              </div>
            </div>
          </div>
        </nav>
        <main className="container mx-auto px-4 py-8">
          {page === 'playground' && <Playground />}
          {page === 'dataset' && <DatasetBrowser />}
          {page === 'evaluation' && <Evaluation />}
          {page === 'components' && <ComponentPlayground />}
        </main>
      </div>
    </QueryClientProvider>
  )
}

export default App
