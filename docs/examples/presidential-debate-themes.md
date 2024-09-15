# Presidential Debate Themes Analysis

This tutorial explains how to analyze themes in presidential debates using the DocETL pipeline. We'll cover the pipeline structure, explain each operation, and discuss the importance of theme resolution.

## Pipeline Overview

Our goal is to build a pipeline that will:

1. Extract key themes and viewpoints from presidential debate transcripts
2. Analyze how these themes have evolved over time, with references to specific debates and quotes

You can take a look at the raw data [here](https://github.com/shreyashankar/docetl/tree/main/example_data/debates/data.json).

Let's examine the pipeline structure and its operations:

```yaml
pipeline:
  steps:
    - name: debate_analysis
      input: debates
      operations:
        - extract_themes_and_viewpoints
        - unnest_themes
        - summarize_theme_evolution
```

??? example "Full Pipeline Configuration"

    ```yaml
    datasets:
      debates:
        type: file
        path: "data.json"

    default_model: gpt-4o

    operations:
      - name: extract_themes_and_viewpoints
        type: map
        output:
          schema:
            themes: "list[{theme: str, viewpoints: str}]"
        prompt: |
          Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

          {{ input.content }}

          Extract the main themes discussed in this debate and the viewpoints of the candidates on these themes.
          Return a list of themes and corresponding viewpoints in the following format:
          [
            {
              "theme": "Theme 1",
              "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
            },
            {
              "theme": "Theme 2",
              "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
            },
            ...
          ]

      - name: unnest_themes
        type: unnest
        unnest_key: themes
        recursive: true

      - name: summarize_theme_evolution
        type: reduce
        reduce_key: theme
        output:
          schema:
            theme: str
            report: str
        prompt: |
          Analyze the following viewpoints on the theme "{{ inputs[0].theme }}" from various debates over the years:

          {% for item in inputs %}
          Year: {{ item.year }}
          Date: {{ item.date }}
          Title: {{ item.title }}
          Viewpoints: {{ item.viewpoints }}

          {% endfor %}

          Generate a comprehensive summary of how Democratic and Republican viewpoints on this theme have evolved through the years. Include supporting quotes from the debates to illustrate key points or shifts in perspective.

          Your summary should:
          1. Identify *all* major trends or shifts in each party's stance over time
          2. Highlight any significant agreements or disagreements between the parties
          3. Note any external events or factors that may have influenced changes in viewpoints
          4. Use specific quotes to support your analysis
          5. The title should contain the start and end years of the analysis

          Format your response as a well-structured report.

    pipeline:
      steps:
        - name: debate_analysis
          input: debates
          operations:
            - extract_themes_and_viewpoints
            - unnest_themes
            - summarize_theme_evolution

      output:
        type: file
        path: "theme_evolution_analysis.json"
        intermediate_dir: "checkpoints"

    ```

## Pipeline Operations

### 1. Extract Themes and Viewpoints

```yaml
- name: extract_themes_and_viewpoints
  type: map
  output:
    schema:
      themes: "list[{theme: str, viewpoints: str}]"
  prompt: |
    Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

    {{ input.content }}

    Extract the main themes discussed in this debate and the viewpoints of the candidates on these themes.
    Return a list of themes and corresponding viewpoints in the following format:
    [
      {
        "theme": "Theme 1",
        "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
      },
      {
        "theme": "Theme 2",
        "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
      },
      ...
    ]
```

This operation processes each debate transcript to identify main themes and candidates' viewpoints. It uses AI to analyze the content and structure the output in a consistent format.

### 2. Unnest Themes

```yaml
- name: unnest_themes
  type: unnest
  unnest_key: themes
  recursive: true
```

The unnest operation flattens the list of themes extracted from each debate. This step prepares the data for further analysis by creating individual entries for each theme.

### 3. Summarize Theme Evolution

```yaml
- name: summarize_theme_evolution
  type: reduce
  reduce_key: theme
  output:
    schema:
      theme: str
      report: str
  prompt: |
    Analyze the following viewpoints on the theme "{{ inputs[0].theme }}" from various debates over the years:

    {% for item in inputs %}
    Year: {{ item.year }}
    Date: {{ item.date }}
    Title: {{ item.title }}
    Viewpoints: {{ item.viewpoints }}

    {% endfor %}

    Generate a comprehensive summary of how Democratic and Republican viewpoints on this theme have evolved through the years. Include supporting quotes from the debates to illustrate key points or shifts in perspective.

    Your summary should:
    1. Identify all major trends or shifts in each party's stance over time
    2. Highlight any significant agreements or disagreements between the parties
    3. Note any external events or factors that may have influenced changes in viewpoints
    4. Use specific quotes to support your analysis
    5. The title should contain the start and end years of the analysis

    Format your response as a well-structured report.
```

This operation analyzes how each theme has evolved over time. It considers viewpoints from multiple debates, identifies trends, and generates a comprehensive summary of the theme's evolution.

## The Need for Theme Resolution

An important consideration in this pipeline is the potential for similar themes to be generated with slightly different wording (e.g., "Climate Change Policy" vs. "Environmental Regulations"). To address this, we need to add a resolve operation before the summarization step.

To synthesize a resolve operation, we can use the `docetl build` command:

```bash
docetl build pipeline.yaml
```

This command adds a resolve operation to our pipeline, resulting in an optimized version:

```yaml
pipeline:
  steps:
    - name: debate_analysis
      input: debates
      operations:
        - extract_themes_and_viewpoints
        - unnest_themes
        - synthesized_resolve_0
        - summarize_theme_evolution
```

The new `synthesized_resolve_0` operation groups similar themes together, ensuring a more accurate and comprehensive analysis of each theme's evolution.

## Running the Optimized Pipeline

With the resolve operation in place, we can now run our optimized pipeline:

```bash
docetl run pipeline_opt.yaml
```

This command processes the debate transcripts, extracts themes, resolves similar themes, and generates summaries of theme evolution over time. The results will be saved in `theme_evolution_analysis.json`, providing insights into the changing landscape of topics discussed in presidential debates. Since we've also set an `intermediate_dir` in our pipeline configuration, intermediate results will be saved in the `intermediate_dir` directory.

Here's the output from running our optimized pipeline:

```bash
$ docetl run workloads/debates/pipeline_opt.yaml
[23:11:08] Performing syntax check on all operations...
           Syntax check passed for all operations.
           Running Operation:
             Type: map
             Name: extract_themes_and_viewpoints
⠴ Running step debate_analysis...
[23:12:48] Intermediate saved for operation 'extract_themes_and_viewpoints'
           Running Operation:
             Type: unnest
             Name: unnest_themes
           Intermediate saved for operation 'unnest_themes'
           Running Operation:
             Type: resolve
             Name: synthesized_resolve_0
[23:12:49] Comparisons saved by blocking: 47689 (98.30%)
⠧ Running step debate_analysis...
[23:13:20] Number of keys before resolution: 312
           Number of distinct keys after resolution: 163
⠴ Running step debate_analysis...
[23:13:22] Self-join selectivity: 0.0288
           Intermediate saved for operation 'synthesized_resolve_0'
           Running Operation:
             Type: reduce
             Name: summarize_theme_evolution
⠇ Running step debate_analysis...
[23:15:56] Intermediate saved for operation 'summarize_theme_evolution'
           Flushing cache to disk...
           Cache flushed to disk.
  Step debate_analysis completed. Cost: $7.37
  Operation extract_themes_and_viewpoints completed. Cost: $5.18
  Operation unnest_themes completed. Cost: $0.00
  Operation synthesized_resolve_0 completed. Cost: $0.03
  Operation summarize_theme_evolution completed. Cost: $2.17
           💾 Output saved to .../theme_evolution_analysis.json
           Total cost: $7.37
           Total time: 287.98 seconds
```

This output shows the progress of our pipeline execution, including the different operations performed, intermediate saves, and the final results. Note the total cost of $7.37 and execution time of about 288 seconds.

## Initial Results

Our pipeline generated reports on various themes discussed in the presidential debates. We've put the results up [here](https://github.com/shreyashankar/docetl/tree/main/example_data/debates/theme_evolution_analysis_baseline.json). However, upon inspection, we found that these reports were lacking in depth and recency. Let's look at a few examples:

!!! example "Example Reports Lacking in Recent Quotes"

    === "Energy"

        ``` markdown
        # Report on Energy Policy Evolution: 1980-2023

        ## Introduction

        This report analyzes the evolution of Democratic and Republican viewpoints on the theme of 'energy' from 1980 to 2023. By examining key debates and exploring the context behind each party's stance, this comprehensive summary highlights the shifts, trends, and defining moments that have shaped energy policy discussions in the United States.

        ## 1980: The Anderson-Reagan Presidential Debate (September 21, 1980)

        ### Democratic Viewpoint (John Anderson)

        In 1980, John Anderson emphasized a new conservation ethic and proposed strategies to reduce dependency on imported oil. He suggested an emergency excise tax on gasoline and a shift in lifestyles to reduce energy consumption, particularly in private automobile use. Anderson's stance highlighted a proactive approach to conservation and a call for significant changes in consumption patterns.

        **Supporting Quote:** "...to convince the American people that we will have to reduce the use of the private automobile."

        ### Republican Viewpoint (Ronald Reagan)

        Ronald Reagan argued that conservation alone wouldn't solve the energy problem and emphasized the potential of untapped energy reserves within the country. He advocated for reducing government restrictions to facilitate energy production and supported nuclear power. Reagan's perspective focused on leveraging the country's existing resources and integrating technological advancements to address energy needs.

        **Supporting Quote:** "...in today's oil wells, there is more oil still there than we have so far taken out and used..."

        ## Trend Analysis

        1. **Democratic Stance on Conservation:** Over time, the Democratic emphasis on energy conservation has evolved. Starting from advocating for reduced automobile use in 1980, this viewpoint has expanded to include broader environmental policies, such as promoting renewable energy sources and reducing carbon emissions.

        2. **Republican Focus on Resource Utilization:** Historically, Republicans have prioritized tapping into existing energy reserves and reducing regulatory barriers. This approach has remained relatively consistent, with an ongoing emphasis on energy independence and support for oil and gas production.

        ## Significant Agreements and Disagreements

        - While both parties have acknowledged the importance of energy security, their methods have often diverged. Democrats have leaned towards conservation and sustainability, while Republicans have focused on resource utilization and deregulation.
        - A notable area of agreement has been the increasing recognition of the role of technology in advancing energy strategies, although the focus of technological investments has varied.

        ## External Influences

        - The oil crises of the 1970s likely influenced the energy conservation efforts of the 1980s, shaping early Democratic policies.
        - Environmental movements and scientific reports on climate change have increasingly driven Democratic policies towards renewable energy and sustainability.
        - Geopolitical events, such as conflicts in oil-producing regions, have reinforced Republican calls for energy independence.

        ## Conclusion

        The evolution of energy policy viewpoints between the Democratic and Republican parties reveals a complex interplay of conservation, resource utilization, technological advancement, and external events.

        Future research should focus on examining more recent debates and policies to provide a comprehensive understanding of current trends and potential future directions. This report serves as a foundation for understanding the historical context and key shifts in energy policy discourse within the United States.
        ```

    === "Racial Profiling"

        ``` markdown
        # Evolution of Democratic and Republican Viewpoints on Racial Profiling (2000-2023)

        ## Introduction

        Racial profiling has been a contentious issue in American politics for decades, with both major parties—Democrats and Republicans—expressing varying perspectives over time. This report analyzes viewpoints from debates over the years, focusing on the evolution of stances, significant agreements and disagreements, and factors influencing changes in perspectives.

        ## Key Trends Over Time

        ### Democratic Viewpoints

        **2000:** In the 2000 Vice Presidential Debate, Senator Joe Lieberman, representing the Democratic party, stated:

        > "Al Gore said ... the first Civil Rights Act legislation we would send to Congress would be a national ban on racial profiling."

        This early stance indicates a clear commitment to addressing racial profiling through legislative action.

        **Key Trends:**
        - Commitment to legislative measures banning racial profiling
        - Strong focus on civil rights and equality

        ### Republican Viewpoints

        **2000:** In the same debate, Dick Cheney, the Republican Vice Presidential candidate, expressed a sympathetic understanding of the anger and frustration caused by racial profiling:

        > "The sense of anger and frustration and rage that go with knowing that the only reason you were stopped, the only reason you were arrested, was because of the color of your skin would make me extraordinarily angry."

        While Cheney acknowledged the emotional impact of racial profiling, his statement did not indicate a specific legislative approach.

        **Key Trends:**
        - Recognition of the emotional and social impact of racial profiling
        - Less emphasis on specific legislative measures compared to Democrats

        ## Agreements and Disagreements

        ### Agreements

        - **Recognition of Issue:** Both parties acknowledge the existence and negative impact of racial profiling.
        - **Emotional Impact:** Both Lieberman and Cheney recognized the profound emotional toll of racial profiling on individuals, although their approaches to addressing it differed.

        ### Disagreements

        - **Legislative Action:** Democrats, represented by Lieberman in 2000, emphasized legislative action to ban racial profiling. In contrast, Cheney's viewpoint, while sympathetic, did not specify any legislative solutions.

        ## Influencing Factors

        The viewpoints on racial profiling have been shaped by various external events and societal changes, including:
        - High-profile cases of police brutality
        - Growing public awareness of systemic racism
        - Movements such as Black Lives Matter

        These factors have pressured both parties to address racial profiling more explicitly in their platforms.

        ## Conclusion

        From 2000 to 2023, the Democratic and Republican parties have shown evolving viewpoints on racial profiling:
        - Democrats have consistently emphasized legislative solutions, reflecting a strong focus on civil rights.
        - Republicans have acknowledged the issue's emotional and social impacts but have been less specific on legislative measures.
        - Notable agreements exist in recognizing the problem, although significant differences remain in approaches to solving it.
        - External events have continuously influenced both parties' stances, pushing them to address racial profiling in their policies.

        ```

    === "Tax Proposals"

        ``` markdown
        # Evolution of Democratic and Republican Tax Proposals (2000-2023)

        ## Introduction
        The debate over tax policy has been a central theme in American politics, featuring prominently in vice-presidential and presidential debates. This report examines the evolution of Democratic and Republican viewpoints on tax proposals from the year 2000 to 2023, highlighting major trends, significant agreements and disagreements, and the impact of external factors.

        ## Democratic Viewpoints: Evolution and Trends

        1. **Early 2000s**: In the 2000 debate, Senator Joe Lieberman emphasized the Democratic focus on saving money for social investments, especially in education, rather than enacting large tax cuts. For instance, Lieberman stated, "We're saving money to invest in education... not going to give it all away in one big tax cut."

        2. **Mid 2000s to Early 2010s**: The Democrats continued to advocate for tax policies that aimed to protect and expand social programs. They argued for targeted tax cuts mainly for middle and lower-income groups, stressing the need for a progressive tax system.

        3. **Late 2010s to Early 2020s**: With the rise of progressive figures within the party, the focus expanded to include more aggressive tax policies on the wealthy and corporations. Proposals such as wealth taxes and increased corporate taxes became more mainstream within the party.

        4. **Present Day**: The Democratic stance has become more nuanced, balancing between the need for social investments and economic recovery post-pandemic. While continuing to advocate for higher taxes on the wealthy, there is a stronger emphasis on ensuring fiscal responsibility and supporting economic growth.

        ## Republican Viewpoints: Evolution and Trends

        1. **Early 2000s**: In the same 2000 debate, Dick Cheney highlighted the Republican view that it was crucial to return surplus revenue to taxpayers through tax cuts. Cheney remarked, "We think it's extraordinarily important... to return it in the form of a tax cut to the American taxpayer."

        2. **Mid 2000s to Early 2010s**: Republicans consistently pushed for broad-based tax cuts, advocating for a reduction in tax rates for both individuals and businesses. They argued that such policies would spur economic growth and job creation.

        3. **Late 2010s to Early 2020s**: The GOP solidified its position on reducing taxes, leading to significant legislative efforts such as the Tax Cuts and Jobs Act of 2017, which lowered taxes for individuals and corporations significantly.

        4. **Present Day**: The focus remains on tax cuts as a means of stimulating economic growth. However, there is increasing attention towards simplifying the tax code and addressing national debt concerns.

        ## Agreements and Disagreements

        - **Agreements**: Both parties have occasionally converged on the necessity of targeted tax relief during economic downturns. For instance, during the COVID-19 pandemic, there was bipartisan support for temporary tax measures to stimulate the economy.

        - **Disagreements**: The primary divide remains on the approach to taxation. Democrats advocate for a more progressive tax structure with higher rates for the wealthy, while Republicans push for across-the-board tax cuts, claiming it benefits economic growth.

        ## Influence of External Factors

        - **Economic Crises**: Events like the 2008 financial crisis and the COVID-19 pandemic have influenced tax policy debates, with both parties adjusting their stances to address economic recovery needs.

        - **Political Shifts**: The rise of progressive movements within the Democratic Party and the consolidation of conservative ideologies within the Republican Party have reshaped tax proposal strategies over the years.

        ## Conclusion
        The tax policy debate between Democrats and Republicans has evolved significantly from 2000 to 2023. While both parties have adjusted their strategies in response to economic conditions and internal political shifts, their core philosophies remain distinct. Through careful analysis of their historical positions and responses to external events, it is evident that tax policy will continue to be a pivotal issue in American politics.
        ```

Upon inspecting the intermediates, it appears that the map operation is doing a good job at extracting relevant information. The issue seems to lie in the reduce operation, which is ignoring some of the analysis.

It's possible that trying to summarize all the insights across all debates for a topic in a single LLM call is too ambitious. To address this, we can set `optimize: true` for the reduce operation.

Let's update our `summarize_theme_evolution` operation in the pipeline:

```yaml
- name: summarize_theme_evolution
  type: reduce
  reduce_key: theme
  optimize: true
  output:
    schema:
      theme: str
      report: str
  prompt: |
    [... existing prompt ...]
```

Rerunning the build command `docetl build pipeline.yaml` will synthesize the optimized reduce operation (make sure `pipeline.yaml` is the pipeline you want to optimize).

## Running the Pipeline (with Optimized Reduce)

In our optimized pipeline, we see that DocETL added a `gleaning` configuration to the reduce operation:

```yaml
- name: summarize_theme_evolution
  type: reduce
  reduce_key: theme
  ...
  gleaning:
    num_rounds: 1
    validation_prompt: |
        1. Does the output adequately summarize the evolution of viewpoints on the theme based on the
        provided debate texts? Are all critical shifts and trends mentioned?
        2. Are there any crucial quotes or data points missing from the output that
        were present in the debate transcripts that could reinforce the analysis?
        3. Is the output well-structured and easy to follow, following any
        formatting guidelines specified in the prompt, such as using headings for
        sections or maintaining a coherent narrative flow?
```

!!! tip

    Note that you should always feel free to edit the `validation_prompt` to better suit your specific needs! The optimizer uses LLMs to write all prompts, but you have the best context on your task and what you're looking for in the output, so you should adjust anything accordingly.

And when running the pipeline, we can observe the impact of this optimization; for example, one of the outputs gets amended to include more recent quotes:

```bash
Validator improvements (gleaning round 1):
1. **Coverage of Critical Shifts and Trends**: While the output captures several significant trends and shifts in Democratic and Republican viewpoints, it could benefit from a more thorough overview, especially about the changing perceptions and proposals related to economic downturns and recoveries across the decades. For instance, it could include how the response to the 2008 financial crisis tied back to historical precedents, linking back to earlier debates about the importance of jobs and middle-class stability (like in the 1976 or 1992 debates).

2. **Missing Quotes and Data Points**: The response could further bolster the analysis by incorporating additional quotes that illustrate the evolving narrative, particularly surrounding tax fairness and job creation. For example, quotes like George Bush in 1984 calling out "working man" perspectives against seemingly progressive taxation could enhance the depth. Additionally, quotes from debates emphasizing the impact of tax cuts on economic recovery and job growth—such as Obama's focus on the automotive industry's recovery in 2012 or McCain's 'putting homeowners first'—could provide essential context to the arguments for both parties.

3. **Structure and Flow**: Overall, the output is fairly well-structured and maintains a logical flow, using headings appropriately to signal key sections. However, it may benefit from clearer subsections under each party's overview to delineate specific key points, such as 'Tax Policy', 'Job Creation', and 'Response to Economic Crises'. This would enhance readability and assist the reader in navigating the shifts in viewpoints. For example, adding bullet points or more vivid transitions between historical periods could clarify the evolution timeline. Moreover, resolving any redundancy (such as multiple mentions of similar themes across years) would streamline the narrative.
```

Check out the new output [here](https://github.com/shreyashankar/docetl/tree/main/example_data/debates/theme_evolution_analysis_reduce_gleaning.json) to see the improvements made by the optimized pipeline! Of course, we can probably optimize the initial map operation too, do prompt engineering, and more to further enhance the pipeline.

!!! note "Interactive Pipeline Creation"

    We're currently building interfaces to interactively create and iterate on these pipelines after seeing outputs. This will allow for more intuitive pipeline development and refinement based on real-time results. If you're interested in this feature or would like to provide input, please reach out to us! Your feedback can help shape the future of DocETL's user experience.
