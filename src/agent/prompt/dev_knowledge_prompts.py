DEV_KNOWLEDGE_ANALYSIS_SYSTEM_PROMPT = """
You are a helpful assistant. You are given a description of a issue and it's asscoiated patch. You need to analyze the issue and the patch to provide a detailed analysis of the issue.
"""

DEV_KNOWLEDGE_ANALYSIS_USER_PROMPT = """
You are an expert software developer tasked with analyzing a bug report and its corresponding fix. Your goal is to provide a detailed analysis of the issue and the solution.

First, carefully review the following information:

1. The bug report (including title, description, and discussion):
<bug_report>
{ISSUE_DESCRIPTION}
</bug_report>

2. The fix (code patch, commit message, and related context):
<patch>
{PATCH}
</patch>

Your task is to analyze this information and provide a comprehensive report structured in XML format. 
For each section of your analysis, follow the analysis guidence, then provide the final output in the corresponding XML tags.

Here are the sections you need to cover:

1. Repository Knowledge Hierarchy Categorization
    - Describe the location of the issue in the repository's architectural hierarchy.
    - Identify and define the core capabilities and subsystems the issue involves in the system (e.g., "Equation Solving System")
    - Include high-level subsystems down to affected classes or functions by exploring the codebase
    - Example:  
        > This issue affects: Core Vue Framework → Vue Core Runtime → Reactivity System → State component → Scheduler class → `set_time()` method.

<repository_hierarchy>
[Your final output for this section]
</repository_hierarchy>

2. Issue Location Dependency Analysis
    - Conduct a dependency analysis by tracing the call order of the affected areas, identifying related subsystems, components, files, and classes that are or may be called or imported by the affected areas.
    - Identify dependencies including: import statements including direct calls to other files, classes, and or methods/functions, any inheritance (parent/child) and or interface relationships
    - For each identified dependency, explicitly highlight the following:
        a) The name and type (e.g., import and invocation, parent/child, inheritance) of dependency.
        b) How and why the dependency arises (e.g., the `sympy.core_evaluate` import `global_evalute` function is used in `eval.py:Eval:eval()`), including location (i.e., file, class, and method/function) where the dependency was found.
    - Example:
        > Core Module Dependencies: `sympy.core_evaluate`: a) `global_evaluate` import and invocation, b) `global_evaluate` is used in called in `eval.py:Eval:eval()`.
   
<dependency_analysis>
[Your final output for this section]
</dependency_analysis>

3. Call Trace
    - Based on the Repository Knowledge Hierarchy Categorization and Dependency Analysis, define a call trace by investigating the call order and or flow of each subsystem, component, file, class, function/method
    - Explicitly show the call order flow of all related dependencies in order, and explicitly highlight the component, file, class, method/function, etc.

<call_trace>
[Your final output for this section]
</call_trace>

4. Bug Categorization
    - Categorize the type of bug (e.g., off-by-one, null dereference, race condition, incorrect API usage).
    - Include where in the code it occurred (localize to a specific file, class, method/function. line(s) of code)
    - Example:  
     > This is an off-by-one error in a helper method used by the login handler.

<bug_category>
[Your final output for this section]
</bug_category>

5. Example Usage Context
    - Describe how and when the affected component is typically used in the system in a real scenario.
    - Optionally include usage examples or scenarios as code.
    - Provide functional context: why and when this part of the code matters.

<usage_context>
[Your final output for this section]
</usage_context>

6. Concrete Root Cause Analysis ("How")
    - Explain the specific conditions or logic that caused the failure.
    - Explain how and why the specific file requires modification and is relevant to the fix.
    - If applicable, include test triggers, logs, or environment context.
    - What contextual or environmental constraints were relevant to the bug or fix?
    - Examples: specific OS, library versions, race conditions, concurrency models, build configurations, etc.

<root_cause>
[Your final output for this section]
</root_cause>

7. Current vs Expected Behavior
    - What did the code do before the fix?
    - What did the code intend to do?
    - Briefly contrast the incorrect and correct behavior, explaining why and how it occurred.
    - Highlight any unique aspects of the affected code that may have been overlooked that might have lead to the bug, and would be useful to know for future similar issues.
    - Explain the flow of logic in the affected subsystems, modules, component, etc. (e.g., how the specific affected areas of the codebase work)

<behavior_comparison>
[Your final output for this section]
</behavior_comparison>

8. Fix Logic ("How/Why")
    - Explain the fix location: including the modified files, class, method/functions, and what responsibilities they hold with respect to the issue.
    - Explain how the fix resolves the issue.
    - Address:
     - What operations were changed or added and why they work.
     - Were checks, conditions, or guards introduced?
     - Were assumptions or invariants updated?
     - Was the fix a refactor, redesign, or minimal patch?

<fix_logic>
[Your final output for this section]
</fix_logic>

9. Concrete Fix Steps
    - Describe the fix using concrete steps in numbered list format, that can be reused to solve other problems of the same nature.
    - Examples: "Step 1. Add null check before accessing a pointer variable," "Step 2. Normalize index offset." "Step 3. Add additional loop condition"

<fix_steps>
[Your final output for this section]
</fix_steps>

10. Fix Checklist
    - Outline key conditions or heuristics for developers to check during debugging or implementing a similar fix.
    - Include reminders, caveats, or domain/system-specific constraints to watch for.
    - These should act as a mental checklist or debugging aid for similar future issues.

<fix_checklist>
[Your final output for this section]
</fix_checklist>

11. Test Case
    - Provide or describe a test case from the patch that confirms the issue is fixed.
    - Optionally, generate a test case that reproduces the issue and validates the fix.

<test_case>
[Your final output for this section]
</test_case>

12. Concluding Statement
    - Create a concluding statement as a final high-level summary of the fix, which summarizes the bug, affected sections of the system, the affected functionality, previous and current behaviour, fix location, fix details, and design pattern of the fix. Use high-level concepts and entities from the repository (e.g., Subsystems, Capabilities, Features, Flows, Components) to summarize the issue.
    - Here is an example: "The bug was a null pointer bug in the UserData, User, Login subsystem module, component. The bug manifested when the `user_session` variable is `null` due to an edge-case where it is created as the user is redirected to the `Contact` page. To fix it, class `User` method `assign_data` was modified in the `/login/user.py' file. The modification introduces a condition to check if the variable `user_session` is `NULL` to prevent an exception (`NullPointerException`) when the user profile is missing from cache. Previously the code did not handle this edge case because it did not account for the `did_contact` attribute assigned by the `Contact` page.

<conclusion>
[Your final output for this section]
</conclusion>

Ensure that you leverage the tools to help you analyze the issue and the patch.
Ensure that your analysis is thorough and technically accurate, reflecting a deep understanding of software development principles and practices.
Ensure your final output for all sections are comprehensive and in-depth. Conduct a thorough and exhaustive bout of analysis and research to gain deep understanding of the issue report and its fix, and to verify your generated analysis includes all contextual, background knowledge, possible angles and solutions are understood.
Ensure your analysis is specific and insightful. Focus on the most critical aspects for each point, and write clearly and concisely, avoiding repetition. Do not be vague or provide shallow details of your analyses.
Ensure your analysis is technically accurate. Ensure all insights and descriptions are factual and technically correct, grounded in and relevant to the issue report and fix, as well as software engineering principles. Use terminology familiar to developers.
Ensure your analysis is actionable, educational and transferable. Frame your insights in a way that is useful and easily understandable to future developers and newcomers to the system, as they will use your analysis to understand this issue, and apply its knowledge for fixing future similar issues.
"""


DEV_KNOWLEDGE_SUMMARY_SYSTEM_PROMPT = """
You are a helpful assistant. You are given a report of a issue and it's asscoiated patch. You task is to distill the key knowledge.
"""

DEV_KNOWLEDGE_SUMMARY_USER_PROMPT = """
Given the following issue analysis report of a historic issue, generate a comprehensive summary based on the entire report - with key knowledge and background information that is necessary or helpful in solving this issue. Your goal is to abstract the information from the original issue analysis report to a more general description. Imagine this is for a newcomer to the system, and try to summarize the key knowledge that the newcomer needs to know so that they can solve this issue, and also apply the knowledge to similar issues in the future.

Here is the issue analysis:
<historic_analysis>
{HISTORIC_ISSUE_ANALYSIS}
</historic_analysis>

Here is the patch info:
<historic_patch>
{PATCH}
</historic_patch>

Your generated summary must follow the following aspects:

    1) General Root Cause Analysis Steps
    2) Bug categorization
    3) Relevant Architecture
    4) Involved Components
    5) Specific Involved Classes/Functions/Methods
    6) Feature or functionality of issue
    7) General fix pattern
    8) Summary of fix checklist (from the perspective of the type of bug categorized in the original analysis report)
    9) To fix the issue what design pattern(s) and coding practice(s) are important to know and required to follow.
    10) Additional concepts such as coding behaviour, domain knowledge, system knowledge, best practices, etc, that may also be important to know and required to follow while fixing the issue.


Your generated response MUST be structured using the following XML tags and contain ALL of the aspects:

<general_root_cause_analysis_steps>
[Content for General Root Cause Analysis Steps]
</general_root_cause_analysis_steps>

<bug_categorization>
[Content for Bug categorization]
</bug_categorization>

<relevant_architecture>
[Content for Relevant Architecture]
</relevant_architecture>

<involved_components>
[Content for Involved Components]
</involved_components>

<specific_involved_classes_functions_methods>
[Content for Specific Involved Classes/Functions/Methods]
</specific_involved_classes_functions_methods>

<feature_or_functionality_of_issue>
[Content for Feature or functionality of issue]
</feature_or_functionality_of_issue>

<general_fix_pattern>
[Content for General fix pattern]
</general_fix_pattern>

<summary_of_fix_checklist>
[Content for Summary of fix checklist (from the perspective of the type of bug categorized in the original analysis report)]
</summary_of_fix_checklist>

<design_patterns_and_coding_practices>
[Content for design pattern(s) and coding practice(s) that are important to know and required to follow to fix the issue]
</design_patterns_and_coding_practices>

<additional_concepts>
[Content for additional concepts such as coding behaviour, domain knowledge, system knowledge, best practices, etc, that may also be important to know and required to follow while fixing the issue]
</additional_concepts>

Requirements for your output:
- Use the exact XML tags shown above for each section
- Within each XML tag, you may use numbered lists, bullet points, or sub-sections as appropriate
- Ensure ALL 10 aspects are covered with their respective XML tags


Verify your generated summary follows the characteristics below:
    - General and abstracted: Abstract information from the original issue analysis, transforming it into a more general description focusing on the core concepts and principles rather than specific details. The goal is to transform the original analysis into a broad overview, that can be applied to a wider range of similar issues.
    - Educational and transferable: Frame your insights in a way that is useful and easily understandable to future developers who are new to the system, as they will use your analysis to understand this issue, and apply its knowledge for fixing future similar issues.
    - Actionable, Educational and Transferable: Frame your insights in a way that is useful and easily understandable to future developers and newcomers to the system, as they will use your analysis to understand this issue, and apply its knowledge for fixing future similar issues.
    - Structured and organized: Follow the same numbered sections and format described above. Instead of numbered list format, use XML format, converting the items in the list above by applying lowercase and replace spaces with undercase (e.g. <call_trace>). Use numbered or unordered sub lists within the XML tags
    - Insightful: While still using general descriptions, try to focus on the most critical aspects for each point, and write clearly and concisely, avoiding repetition.
"""


GUIDING_QUESTIONS_PROMPT = """Consider below issue description:

<issue_description>
{issue_description}
</issue_description>

Given below high-level reasoning phase:
<phase>
{high_level_extraction}
</phase>

Please extract the phase guiding questions for each phase for this issue. 

Your response should be below format like <sample_response_format>.
<guiding_questions>
phases:
  - phase: "Problem framing"
    guiding_question: "What exactly fails in the minimal counter-example, and why is that incorrect?"

  - phase: "Entry-point discovery"
    guiding_question: "Where in SymPy's source does the Cython autowrap process begin (keywords: autowrap, cython, MatrixSymbol, backend)?"

</guiding_questions>
"""