#!/bin/bash

# Ralph - Autonomous AI Agent Loop
# Runs AI coding tools repeatedly until all PRD items are complete
# Based on https://ghuntley.com/ralph/

set -e

# Configuration
TOOL="amp"  # Default to amp
MAX_ITERATIONS="10"
PRD_FILE="prd.json"
PROGRESS_FILE="progress.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                MAX_ITERATIONS="$1"
            fi
            shift
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
check_prerequisites() {
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq is required but not installed${NC}"
        echo "Install with: brew install jq"
        exit 1
    fi

    if ! command -v "$TOOL" &> /dev/null; then
        echo -e "${RED}Error: $TOOL command not found${NC}"
        exit 1
    fi

    if [ ! -f "$PRD_FILE" ]; then
        echo -e "${RED}Error: $PRD_FILE not found${NC}"
        exit 1
    fi

    if [ ! -f "prompt.md" ]; then
        echo -e "${RED}Error: prompt.md not found${NC}"
        exit 1
    fi
}

# Find the next story to work on
get_next_story() {
    jq -r '.userStories[] | select(.passes == false) | .id' "$PRD_FILE" | head -1
}

# Check if all stories are complete
all_stories_complete() {
    local incomplete=$(jq -r '.userStories[] | select(.passes == false) | .id' "$PRD_FILE" | wc -l)
    [ "$incomplete" -eq 0 ]
}

# Get story details
get_story_details() {
    local story_id=$1
    jq --arg id "$story_id" '.userStories[] | select(.id == $id)' "$PRD_FILE"
}

# Update story status
update_story_status() {
    local story_id=$1
    local status=$2
    jq --arg id "$story_id" --arg status "$status" \
        '(.userStories[] | select(.id == $id) | .passes) |= ($status == "true")' \
        "$PRD_FILE" > "${PRD_FILE}.tmp" && mv "${PRD_FILE}.tmp" "$PRD_FILE"
}

# Initialize progress file
init_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        cat > "$PROGRESS_FILE" << 'EOF'
# Ralph Progress Log

This file tracks learnings and context across Ralph iterations.
Each iteration appends findings here so future iterations can benefit.

## Session Start

EOF
    fi
}

# Run one iteration autonomously
run_iteration() {
    local iteration=$1
    local story_id=$2
    local story_details=$3

    local story_title=$(echo "$story_details" | jq -r '.title')
    local priority=$(echo "$story_details" | jq -r '.priority')
    local description=$(echo "$story_details" | jq -r '.description')
    local acceptance_criteria=$(echo "$story_details" | jq -r '.acceptanceCriteria[]' | sed 's/^/- /')

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Iteration ${iteration}/${MAX_ITERATIONS}${NC}"
    echo -e "${BLUE}Story: ${story_id} (priority ${priority})${NC}"
    echo -e "${BLUE}Title: ${story_title}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Create temp file for results
    local result_file="/tmp/ralph-result-${story_id}.txt"
    rm -f "$result_file"

    # Create the task for the AI
    cat > /tmp/ralph-task.md << TASK_EOF
# Task: ${story_id}

## Story
${story_title}

## Description
${description}

## Acceptance Criteria
${acceptance_criteria}

## Instructions

You are an AI implementing a user story. Your task:

1. **Implement** all acceptance criteria for this story
2. **Run quality gates**:
   - \`mise run lint\` (must pass)
   - \`mise run test\` (must pass)
   - \`mise run verify\` (must pass)
3. **Report results** in this exact format at the end:

RALPH_RESULT_START
{
  "story_id": "${story_id}",
  "passed": true/false,
  "errors": ["error message 1", "error message 2"],
  "summary": "Brief summary of what was implemented"
}
RALPH_RESULT_END

## Quality Gates

Before reporting success (passed: true):
- All acceptance criteria are implemented
- Code passes lint (no style/import errors)
- Tests pass (all test suites green)
- Type checking passes (no TypeScript/Python errors)
- No obvious bugs or incomplete code

If any quality gate fails, report passed: false with error details.

## Context

Refer to prompt.md for full project context and known patterns.
Read AGENTS.md for codebase conventions and common gotchas.

## Additional Context from Previous Iterations

$(tail -100 progress.txt 2>/dev/null || echo "No previous iterations")

TASK_EOF

    # Invoke the AI tool
    echo -e "${YELLOW}Invoking ${TOOL}...${NC}"
    
    if [ "$TOOL" = "amp" ]; then
        # Use Amp with task file
        amp < /tmp/ralph-task.md > /tmp/ralph-output.txt 2>&1 || true
    else
        # Use Claude Code
        $TOOL code < /tmp/ralph-task.md > /tmp/ralph-output.txt 2>&1 || true
    fi

    # Parse results from output
    if grep -q "RALPH_RESULT_START" /tmp/ralph-output.txt; then
        local json_result=$(sed -n '/RALPH_RESULT_START/,/RALPH_RESULT_END/p' /tmp/ralph-output.txt | sed '1d;$d')
        local passed=$(echo "$json_result" | jq -r '.passed' 2>/dev/null || echo "false")
        local summary=$(echo "$json_result" | jq -r '.summary' 2>/dev/null || echo "Implementation completed")
        local errors=$(echo "$json_result" | jq -r '.errors[]?' 2>/dev/null)

        if [ "$passed" = "true" ]; then
            echo -e "${GREEN}✓ Story ${story_id} PASSED${NC}"
            echo "  Summary: $summary"
            update_story_status "$story_id" "true"
            git add -A
            git commit -m "[${story_id}] ${story_title}"
            
            # Append to progress
            echo "" >> "$PROGRESS_FILE"
            echo "## Iteration ${iteration} - ${story_id}" >> "$PROGRESS_FILE"
            echo "" >> "$PROGRESS_FILE"
            echo "**Status**: ✓ PASSED" >> "$PROGRESS_FILE"
            echo "**Title**: ${story_title}" >> "$PROGRESS_FILE"
            echo "**Summary**: ${summary}" >> "$PROGRESS_FILE"
            echo "" >> "$PROGRESS_FILE"
            
            return 0
        else
            echo -e "${YELLOW}✗ Story ${story_id} FAILED${NC}"
            echo "  Summary: $summary"
            if [ ! -z "$errors" ]; then
                echo "  Errors:"
                echo "$errors" | while read err; do
                    echo "    - $err"
                done
            fi
            
            # Append to progress
            echo "" >> "$PROGRESS_FILE"
            echo "## Iteration ${iteration} - ${story_id} (RETRY)" >> "$PROGRESS_FILE"
            echo "" >> "$PROGRESS_FILE"
            echo "**Status**: ✗ FAILED" >> "$PROGRESS_FILE"
            echo "**Title**: ${story_title}" >> "$PROGRESS_FILE"
            echo "**Summary**: ${summary}" >> "$PROGRESS_FILE"
            if [ ! -z "$errors" ]; then
                echo "**Errors**:" >> "$PROGRESS_FILE"
                echo "$errors" | while read err; do
                    echo "  - $err" >> "$PROGRESS_FILE"
                done
            fi
            echo "" >> "$PROGRESS_FILE"
            
            return 1
        fi
    else
        # Could not parse result - treat as failure
        echo -e "${RED}✗ Story ${story_id} - Could not parse AI output${NC}"
        echo "AI output:" >> "$PROGRESS_FILE"
        tail -20 /tmp/ralph-output.txt >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"
        return 1
    fi
}

# Main loop
main() {
    echo -e "${GREEN}Ralph - Autonomous Agent Loop${NC}"
    echo ""

    check_prerequisites
    init_progress

    # Create branch if needed
    if ! git rev-parse --verify improve-training-data &> /dev/null 2>&1; then
        echo -e "${YELLOW}Creating branch: improve-training-data${NC}"
        git checkout -b improve-training-data
    else
        git checkout improve-training-data
    fi

    local iteration=1
    while [ "$iteration" -le "$MAX_ITERATIONS" ]; do
        if all_stories_complete; then
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${GREEN}✓ ALL STORIES COMPLETE!${NC}"
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo -e "${GREEN}<promise>COMPLETE</promise>${NC}"
            echo ""
            break
        fi

        local next_story=$(get_next_story)
        if [ -z "$next_story" ]; then
            echo -e "${GREEN}No more stories to complete${NC}"
            break
        fi

        local story_details=$(get_story_details "$next_story")
        run_iteration "$iteration" "$next_story" "$story_details"

        ((iteration++))
    done

    if [ "$iteration" -gt "$MAX_ITERATIONS" ]; then
        echo -e "${YELLOW}Maximum iterations (${MAX_ITERATIONS}) reached${NC}"
        echo -e "${YELLOW}Some stories may still be incomplete${NC}"
        echo ""
        echo "To continue, run: ./ralph.sh --tool $TOOL $((MAX_ITERATIONS + 10))"
    fi
}

main
