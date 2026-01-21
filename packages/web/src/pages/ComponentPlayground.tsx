/**
 * Component Playground - Visual testing for UI components in isolation.
 * Use Chrome DevTools MCP to verify components render and function correctly.
 */

import { useState } from 'react'
import { CopyButton } from '@/components/CopyButton'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function ComponentPlayground() {
  const [lastAction, setLastAction] = useState<string>('')

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Component Playground</h1>
        <p className="text-muted-foreground">
          Visual testing for UI components. Use Chrome DevTools MCP to verify.
        </p>
      </div>

      {lastAction && (
        <div className="p-3 bg-green-100 text-green-800 rounded-md" data-testid="last-action">
          Last action: {lastAction}
        </div>
      )}

      {/* CopyButton Component */}
      <Card>
        <CardHeader>
          <CardTitle>CopyButton</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Copies text to clipboard. Shows checkmark on success.
          </p>

          <div className="space-y-2">
            <div className="flex items-center gap-2 p-2 bg-muted rounded">
              <code className="flex-1">repo:github.com/example lang:python</code>
              <CopyButton
                text="repo:github.com/example lang:python"
                data-testid="copy-button-1"
              />
            </div>

            <div className="flex items-center gap-2 p-2 bg-muted rounded">
              <code className="flex-1">type:commit author:john after:2024-01-01</code>
              <CopyButton
                text="type:commit author:john after:2024-01-01"
                data-testid="copy-button-2"
              />
            </div>
          </div>

          <div className="text-sm">
            <strong>Test:</strong> Click copy button → icon should change to checkmark → reverts after 2s
          </div>
        </CardContent>
      </Card>

      {/* Button Variants */}
      <Card>
        <CardHeader>
          <CardTitle>Button Variants</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => setLastAction('Default button clicked')}>
              Default
            </Button>
            <Button variant="outline" onClick={() => setLastAction('Outline button clicked')}>
              Outline
            </Button>
            <Button variant="ghost" onClick={() => setLastAction('Ghost button clicked')}>
              Ghost
            </Button>
            <Button disabled>Disabled</Button>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button size="sm">Small</Button>
            <Button size="default">Default</Button>
            <Button size="lg">Large</Button>
          </div>
        </CardContent>
      </Card>

      {/* Interactive Test Area */}
      <Card>
        <CardHeader>
          <CardTitle>Interactive Test</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Click buttons and verify the "Last action" message updates correctly.
          </p>

          <div className="flex gap-2">
            <Button onClick={() => setLastAction('Action A triggered at ' + new Date().toLocaleTimeString())}>
              Trigger Action A
            </Button>
            <Button variant="outline" onClick={() => setLastAction('Action B triggered at ' + new Date().toLocaleTimeString())}>
              Trigger Action B
            </Button>
            <Button variant="ghost" onClick={() => setLastAction('')}>
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
