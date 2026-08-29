# Exemplars for Personal Health Insights Agent (PHIA) Capabilities

These notebooks are parsed and converted into exemplars for a ReActAgent. Each
notebook should follow the ReAct framework, adhering to the following format:

*   The first cell contains the problem statement.
*   The second cell describes the initial thought process or context for solving
    the problem.
*   Subsequent cells follow this structure:
    -   A text cell describing a thought or rationale behind the next action
        (aligned with the ReAct framework's Thought step).
    -   A code cell implementing the idea or action described in the previous
        cell (aligned with the ReAct framework's Act step).
    -   A result or observation from evaluating the previous code cell (aligned
        with the ReAct framework's Observe step).
*   The final cell provides a concluding thought and any final outputs or
    insights (aligned with the ReAct framework's Finish step).

This format ensures that each ReActStep except for the final one with
`is_finished` as True has a Thought, Action, and Observation. The final step has
a Thought, Action, and an Observation set to None.
