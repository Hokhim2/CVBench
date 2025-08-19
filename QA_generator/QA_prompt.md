I would like you to act as a quizmaster who designs questions based on provided captions of each videos and summary of the interconnections between videos that should be grounded in real-life contexts and made more relevant to everyday realities. You are provided with individual video captions (per timestamps) and a summary of the interconnections between the videos.

You are to generate relevant Question-Answer (QA) pairs by analyzing key interconnections between videos while also referring to the detailed video summary. Your questions should specifically target the interconnections between the videos. Your question should be answerable, and checkable.

These QAs should be categorized with the following task definitions:

1. **Cross-video Anomaly Detection**:

   - Concept: the comparative analysis of multi-source video data identifies anomalous segments deviating from normal patterns through global spatiotemporal feature and behavioral pattern comparisons, while pinpointing their underlying causes.

   - Difference: by transcending the limitations of single-video local perspectives, analyze anomaly saliency based on the richness of multi-video events and the coherence of events from multiple viewpoints.

   - Example questions:

     - "What noteworthy thing has happened in the listed videos that deserves attention?" 

       "A. Shared bicycles are in chaos.", "B. Pedestrians robbed.", "C. Bus driver gets off.", "D. Stores are closed."
       
     - "Which of the listed videos contains an anomaly compared to the others?" 
     
       "A. Video 1.", "B. Video 2.", "C. Video 3.", "D. Video 4."

2. **Cross-video Scene Recognition**:

   - Concept: this category involves the fusion of semantic features from multiple videos to quickly locate scene segments that match the target spatiotemporal attributes.

   - Difference: it focuses on identifying differences across multiple videos rather than similarities within a single video.

   - Example questions:

     - "Is Wednesday the day when the dog was found missing from the following 4 days of surveillance?"

       "Yes.", "No."
       
     - "What scene is covered across all four listed videos?" 
     
       "A. Military base.", "B. Post-disaster scene.", "C. Rescue team.", "D. Overcrowded supermarkets."

3. **Multi-video Key-Action Recognition**:

   - Concept: this category involves spatiotemporal alignment of motion trajectories across videos to identify differences in action execution, supporting action quality assessment and optimization.

   - Difference: it emphasizes modeling the spatiotemporal consistency of actions across multiple videos, rather than classifying actions within a single video.

   - Example questions:

     - "Which action is the biggest difference between my Tai Chi (video 1) and the standard Tai Chi (video 2), and needs the most improvement?"

       "A. Beginning Posture.", "B. White Crane Spreads Its Wings.", "C. Play the Lute.", "D. Single Whip."
       
     - "What action did both videos do to improve shooting accuracy?" 
     
       "A. Wear gloves.", "B. Lean against the wall.", "C. Add Grip.", "D. Carrying a backpack."

4. **Cross-video Event Retrieval**:

   - Concept: this category involves rapidly locating segments across multiple videos that meet specific event elements, enabling effective filtering and selection.

   - Difference: the events are distributed across different videos, rather than being contained within a single video.

   - Example questions:

     - "Which of the following videos is a recipe for Sichuan-style spicy chicken?"

       "A. Video 1.", "B. Video 2.", "C. Video 3.", "D. Video 4."

5. **Cross-video Object Recognition**:

   - Concept: this category involves fusing multi-view object features to address challenges such as occlusion and deformation, enabling consistent identity recognition of objects across videos.

   - Difference: it leverages multi-source information to compensate for the limitations of single-video perspectives.

   - Example questions:

     - "What is the black and white animal that fights with the cat?"

       "A. Skunk.", "B. Dog.", "C. Raccoon.", "D. Panda."

6. **Multi-video Attribute Recognition**:

   - Concept: this category involves confirming and extracting the attributes (such as texture, color, function, relationship, characteristic and shape) of a specific target across multiple videos, capturing the same target in different states.

   - Difference: it focuses on cross-video attribute state transitions, rather than static attribute recognition within a single frame.

   - Example questions:

     - "What color was the final cake?"

       "A. Green.", "B. White.", "C. Black.", "D. Pink."
       
     - "what is the function of aluminum foil?"
     
       "A. Contain.", "B. Shape.", "C. Enhance flavor.", "D. Change color."

7. **Joint-Video Counting**:

   - Concept: this category involves the precise identification and statistical analysis of the same target across multiple videos.

   - Difference: the targets are distributed across multiple videos rather than a single video, requiring cross-video unified authentication of the targets, while eliminating limitations of single-video perspectives (such as occlusion). Targets may also be abstract, such as events.

   - Example questions:

     - "How many total activities did he do over the summer?"

       "A. 1.", "B. 2.", "C. 3.", "D. 4."
       
     - "How many basketballs are there in the whole house?"
     
       "A. 1.", "B. 2.", "C. 3.", "D. 4."

8. **Cross-Video Entity Matching**:

   - Concept: this category involves making similarity judgments of entities across multiple videos with varying spatiotemporal conditions (different space and time, same space and different times, different space and different times), such as identifying criminal suspects.

   - Difference: it emphasizes identity association across non-overlapping fields of view, overcoming the limitations of single-view tracking.

   - Example questions:

     - "The following is a complete video from different perspectives and in random order, which of the same person appearing in different perspectives?"

       "A. Child in blue.", "B. Adult in blue.", "C. Adult in red.", "D. Child in red."

9. **Multi-View Scene Understanding**:

   - Concept: this category involves integrating spatiotemporal clues from multiple perspectives to reconstruct the complete causal chain and logical relationships of event development.

   - Difference: it constructs a global event knowledge graph, rather than understanding event fragments from a single perspective.

   - Example questions:

     - "According to the videos listed, why is he angry?"
   
       "A. He had bad luck from morning to afternoon.", "B. His family was robbed.", "C. He quarreled with someone over the morning incident.", "D. He's angry on purpose."
       
     - "According to the videos listed, how to make a cake?"
     
       "A. Process the ingredients - ferment - place into the mold - steam.", "B. Ferment - process the ingredients - place into the mold - steam.", "C. Process the ingredients - place into the mold - steam - ferment.", "D. Place into the mold - process the ingredients - ferment - steam."
   
10. **Multi-Video Temporal Reasoning**:

    - Concept: this category involves integrating multiple videos and making judgments about hidden logical relationships at specific times, such as predicting the future or sorting events.

    - Difference: it addresses the asynchronous timestamp issue in single videos, enabling global temporal modeling.

    - Example questions:

      - "The listed videos are temporal disorganized; what is the correct video order?"

        "A. 1-2-3-4.", "B. 2-1-4-3.", "C. 3-2-4-1.", "D. 4-3-2-1."

      - "Based on the videos listed, what is he most likely to do next?"

        "A. Wash clothes.", "B. Going out to buy groceries.", "C. Wash dishes.", "D. Play games."

11. **Joint-Video Spatial Navigating**:

    - Concept: this category involves fusing multi-view geometric information to construct a 3D spatial semantic map, supporting cross-view path planning.

    - Difference: spatial registration and joint reasoning of multi-source visual data.

    - Example questions:

      - "The following is a complete video from different perspectives and in random order, if sitting on a large chair close to the television, how to get to the stove?"

        "A. Head to the top-right corner to enter the kitchen and stove is located in the top-right corner of the kitchen.", "B. Head to the top-left corner to enter the kitchen and stove is located in the top-left corner of the kitchen.", "C. Head to the top-right corner to enter the kitchen and stove is located in the top-left corner of the kitchen.", "D. Head to the top-left corner to enter the kitchen and stove is located in the top-right corner of the kitchen."

12. **Video Difference Caption**:

    - Concept: this category involves fine-grained cross-video comparison, identifying differences across multiple videos in dimensions such as event progression and object states.

    - Difference: it emphasizes the identification of differences in dynamic processes, rather than static image comparison.

    - Example questions:

      - "What is the difference in the videos listed?"

        "A. The core message the video intends to convey.", "B. Narrative method.", "C. Emotional atmosphere.", "D. None of the above.. "

13. **Cross-video Counterfactual Reasoning**:

    - Concept: based on the spatiotemporal factual foundation from multiple videos, this category constructs a causal inference chain for a virtual scenario.

    - Difference: causal effect estimation based on multi-source observational data, overcoming the limitations of single-video causal inference.

    - Example questions:

      - "What wouldn't have happened if the black man had gone straight home?"

        "A. Gas explosion.", "B. Fire.", "C. Friends returned home.", "D. Confiscated electricity bills."

14. **Joint-Video Summarization**:

    - Concept: this category involves extracting semantic information from multiple videos to generate event logic and descriptions that satisfy specific query requirements.

    - Difference: cross-video information fusion and semantic compression, rather than simple fragment stitching.

    - Example questions:

      - "Summarize the event with the listed fragmented videos."

        "A. A student helped an old woman who fell, but she later accused him of causing it, leading to family conflict.", "B. A student ignored a fallen elderly woman, which led to guilt and family tension.", "C. A student rescued an elderly woman from danger and was honored by the community.", "D. A student had a misunderstanding with an elderly woman, but they reconciled and became friends."
    
15. **Cross-Video Procedural Transfer**:

    - Concept: this category focuses on cross-video dynamic knowledge transfer capabilities, this approach analyzes the knowledge in the source video (such as first aid steps or mathematical problem solving methods) and adaptively refines or supplements the execution logic of similar tasks in the target video, while overcoming the interference caused by scene differences.

    - Difference: associating procedural semantics across videos, as a single video can only capture isolated processes and cannot achieve knowledge generalization or adaptation.

    - Example questions:

      - "Based on the first aid knowledge in video 1, what should be done to save the person drowning in video 2?"

        "A. Call for help immediately, use a floating object to assist, and perform CPR if needed.", "B. Jump into the water right away to pull the person out as quickly as possible.", "C. Wait until the person stops moving, then drag them to shore and check for a pulse.", "D. Let others handle it, as attempting rescue without training could be dangerous."

 Guidelines for Response:

1. The questions can be set into two types: multiple-choice questions (containing four options where only one correct answer is given and the others are confusing and incorrect, do not give only one correct option), and yes-no questions (only yes or no answers are required). If descriptions are ambiguous, (e.g., timeline or content), prioritize answer uniqueness. Please remember that multiple-choice questions must have four options, and yes-no questions have two options.
2. Avoid excessive similarity between options.
3. When involving multiple subjects, use subject attributes (e.g.,“adult in red”) for reference. Avoid abstract identifiers like “subject 1” or “person A”.
4. Different questions should cover more videos content,avoiding repetitive questioning. 
4. Questions should avoid sentences such as "based on captions" or "baesd on descriptions" indicating that the questions is derived from captions.
5. Avoid questions that may prompt to seek solutions from specific videos, the questions should focus on cultivating a comprehensive understanding and critical thinking across all videos, ensuring that answers can only be derived after viewing the complete content series.
6. The questions should be focused on the interconnections between the videos, including differences, correlations, matches, etc. Questions should avoid absolute timestamps and absolute spatial locations and use relative terms. For example, after drinking water, next to the TV, etc.
7. You should try to be innovative, and you may increase the difficulty of the questions, the questions should avoid being simple, but the questions should be grounded in real-life contexts and made more relevant to everyday realities.

Output Format: 

Your entire response must be formatted in JSON as shown below:

{ 

 "QA_pairs": [  

 {"Task type": "", "Question": "","Options": ["A. ", "B. ", "C. ", "D. "], Answer": "A"},  

 {"Task type": "", "Question": "","Options": ["A. ", "B. ", "C. ", "D. "], Answer": "B"},  

 {"Task type": "", "Question": "","Options": ["Yes. ", "No. "], Answer": "Yes"}, 

  {"Task type": "", "Question": "","Options": ["A. ", "B. ", "C. ", "D. "], Answer": "D"}  ]

 }

The task types are the 15 types above, and each type can be freely set as multiple-choice or yes-no question. Instead of strictly adhering to designing a question for each type, you can design more questions for task types you think are appropriate, or no questions for those that are not, thus ensuring the quality of the questions.
