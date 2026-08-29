# Domain Expert Agent Few-Shot Examples

This folder contains Jupyter notebook-based exemplars for the Domain Expert Agent.
These exemplars teach the model the correct ReAct format for tool usage.

## Format

Each notebook follows the PHIA format:
1. First cell (markdown): The question/prompt
2. Subsequent cells alternate between:
   - Markdown: Thought/reasoning
   - Code: Tool usage or final answer

### Tool Usage Format

**Search tool:**
```python
# search('your search query here')
```
The search query is extracted from within the single quotes.

**tool_code (Python execution):**
```python
# Actual Python code goes here
some_function_call()
```

**Final answer:**
```python
# Final answer
print("""Your comprehensive response here""")
```

## Exemplars Included

1. `cholesterol_interpretation.ipynb` - LDL/HDL/Total cholesterol interpretation
2. `hba1c_prediabetes.ipynb` - HbA1c and diabetes risk assessment
3. `blood_pressure.ipynb` - Blood pressure classification
4. `liver_enzymes.ipynb` - ALT/AST liver function
5. `kidney_function.ipynb` - Creatinine/eGFR kidney assessment
6. `thyroid_tsh.ipynb` - TSH and thyroid function
7. `vitamin_d.ipynb` - Vitamin D deficiency
8. `improve_heart_health.ipynb` - General cardiovascular health advice
9. `iron_anemia.ipynb` - Ferritin/hemoglobin and iron deficiency anemia
10. `metabolic_summary.ipynb` - Metabolic panel summary with reference ranges
11. `rhr_hrv.ipynb` - Resting heart rate and HRV interpretation

## Adding New Exemplars

To add a new exemplar:
1. Create a new `.ipynb` file following the format above
2. Ensure the first cell contains the question
3. Use `# search('query')` for search tool calls
4. Use regular Python code for tool_code calls
5. End with a `print()` statement for the final answer
