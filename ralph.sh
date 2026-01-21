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

    if ! command -v amp &> /dev/null && ! command -v claude &> /dev/null; then
        echo -e "${RED}Error: Neither amp nor claude command found${NC}"
        echo "Install one of:"
        echo "  - Amp: https://ampcode.com"
        echo "  - Claude Code: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi

    if [ ! -f "$PRD_FILE" ]; then
        echo -e "${RED}Error: $PRD_FILE not found${NC}"
        echo "Create it with: amp skill prd"
        exit 1
    fi

    if [ ! -f "prompt.md" ]; then
        echo -e "${RED}Error: prompt.md not found${NC}"
        echo "This file is required for Ralph to work"
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
        echo "Initialized $PROGRESS_FILE"
    fi
}

# Run one iteration
run_iteration() {
    local iteration=$1
    local story_id=$2
    local story_details=$3

    local story_title=$(echo "$story_details" | jq -r '.title')
    local priority=$(echo "$story_details" | jq -r '.priority')

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Iteration ${iteration}/${MAX_ITERATIONS}${NC}"
    echo -e "${BLUE}Story: ${story_id} (priority ${priority})${NC}"
    echo -e "${BLUE}Title: ${story_title}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Create context for the AI
    local ai_context="
# Current Task

Story ID: ${story_id}
Title: ${story_title}
Priority: ${priority}

## Story Details
$(echo "$story_details" | jq -r '.description')

## Acceptance Criteria
$(echo "$story_details" | jq -r '.acceptanceCriteria[]' | sed 's/^/- /')

## Notes
$(echo "$story_details" | jq -r '.notes')

---

Read prompt.md for full context about this project and quality gates.
"

    # Run the appropriate tool
    if [[ "$TOOL" == "claude" ]]; then
        # Claude Code
        if command -v claude &> /dev/null; then
            echo -e "${YELLOW}Invoking Claude Code...${NC}"
            claude code << 'CLAUDE_EOF'
$ai_context

Please implement this story. Refer to prompt.md for quality gates and requirements.
When done, let me know by saying: STORY_COMPLETE
CLAUDE_EOF
        else
            echo -e "${RED}Error: claude command not found${NC}"
            return 1
        fi
    else
        # Amp (default)
        if command -v amp &> /dev/null; then
            echo -e "${YELLOW}Invoking Amp...${NC}"
            amp << 'AMP_EOF'
$ai_context

Please implement this story. Refer to prompt.md for quality gates and requirements.
When done, confirm all acceptance criteria are met.
AMP_EOF
        else
            echo -e "${RED}Error: amp command not found${NC}"
            return 1
        fi
    fi

    echo ""
    echo -e "${YELLOW}Did story ${story_id} pass all acceptance criteria? (y/n)${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        update_story_status "$story_id" "true"
        git add -A
        git commit -m "[${story_id}] ${story_title}"
        
        # Append to progress
        echo "" >> "$PROGRESS_FILE"
        echo "## Iteration ${iteration} - ${story_id}" >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"
        echo "**Completed**: ${story_title}" >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"
        
        echo -e "${GREEN}✓ Story ${story_id} marked as complete${NC}"
        return 0
    else
        echo -e "${YELLOW}Story ${story_id} not yet complete. Review feedback and try again.${NC}"
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
    fi
}

main
