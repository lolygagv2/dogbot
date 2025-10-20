# Safe Cleanup Command

Before any cleanup:
1. List ALL files that will be affected
2. Categorize into: keep, archive, delete
3. Show directory structure before/after
4. Wait for explicit "yes" to proceed
5. Move files to /tests/archive/ not permanent deletion

Never delete:
- notes.txt
- Any .md files in /docs/
- config.yaml
- Files modified in last 7 days without permission
```

Then use: `/project:safe-cleanup`

## **Specific Prompt for Your Current Mess**

Use this exact prompt in Plan Mode:
```
I need to organize this project. Follow these rules from CLAUDE.md:

1. INVENTORY PHASE:
   - List ALL Python files outside /tests/ directory
   - List ALL image files outside standard locations (vision/, docs/)
   - Identify duplicate or similar test files
   
2. CATEGORIZATION:
   Show me files organized by:
   - Keep (active development)
   - Archive (old tests, might need later)
   - Delete candidates (failed experiments, duplicates)

3. PROPOSED STRUCTURE:
   /tests/
     /hardware/  - hardware test files
     /ai/        - AI/vision test files  
     /audio/     - audio test files
     /integration/ - integration tests
     /archive/   - old test files (dated folders)

4. Show me the reorganization plan with:
   - File count in each category
   - Before/after directory tree
   - Which files you're unsure about

DO NOT EXECUTE - Wait for my approval on each category.