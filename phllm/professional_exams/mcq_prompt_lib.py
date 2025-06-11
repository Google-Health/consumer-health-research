# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constants related to prompts used in  PHLLM project."""

VALID_ANSWER_OPTIONS = [f'({letter})' for letter in 'ABCDEF']

MCQS_EVAL_INSTRUCTION = (
    'Instruction: The following is a multiple choice question about {domain}'
    ' knowledge. Output a single option from {options} as the final answer.\n'
)

VALID_HEALTH_DOMAINS = ['Sleep', 'Fitness']

SLEEP_COT_MCQ = (
    'Instructions: The following are multiple choice questions about {domain}'
    ' knowledge. Solve them in a step-by-step fashion, starting by summarizing'
    ' the available information. Output a single option from {mcq_options} as'
    ' the final answer and enclosed by xml tags <answer></answer>.\n\n Here are'
    ' two examples:\n ##Question: A 26-year-old female presents asking about'
    ' jet lag. She has no past medical history, lives on the East Coast, and'
    " travels frequently to the West Coast for business. The person's career"
    ' involves planning evening events, and she reports significant sleepiness'
    ' at these events that impairs her ability to perform her job. She wants to'
    ' know how she can adapt to Pacific Standard Time (PST) before she travels.'
    ' What treatment plan will help this patient adapt to PST prior to'
    ' travel?\n(A) Light in evening and later bedtime 1 day before'
    ' traveling\n(B) Light in morning and earlier wake time 3 days before'
    ' traveling\n(C) Light in evening and later bedtime 3 days before'
    ' traveling\n(D) Light in morning and earlier wake time 1 month before'
    ' traveling\n(E) Light in evening and later bedtime 1 month before'
    " traveling\nExplanation: Let's solve this step-by-step, referring to"
    ' authoritative sources as needed. The West Coast is 3 timezones behind the'
    ' East Coast. Since she plans evening events, she needs to shift her'
    ' schedule to stay up 3 hours later. Adding light in the evening will'
    ' disrupt melatonin production, delaying sleepiness. Transitioning'
    ' timezones typically takes one day per timezone.\nAnswer:'
    ' <answer>(C)</answer>\n\n##Question: What is a difference in the clinical'
    ' features of obstructive sleep apnea (OSA) in older adults compared to'
    ' younger adults?\n(A) Increased prevalence of OSA among older adults'
    ' occurs after age 65.\n(B) Clinical symptoms associated with OSA (e.g.'
    ' excessive daytime sleepiness) are less common and less severe in older'
    ' adults than in younger adults.\n(C) Increased risk of cardiopulmonary'
    ' diseases is greater among elderly than among younger individuals.\n(D)'
    ' Excess body weight, snoring, and witnessed apneas more consistently'
    ' indicate OSA in older adults than in younger individuals.\n(E) There are'
    ' no significant OSA differences between older and younger'
    " adults.\nExplanation: Let's solve this step-by-step, referring to"
    ' authoritative sources as needed. Compared to younger patients with the'
    ' same apnea hypopnea index, OSA in older patients is associated with less'
    ' sleepiness (Morrell et al 2012). This observation has led some to suggest'
    ' that OSA in the elderly may represent a distinct physiological'
    ' phenotype.\nAnswer: <answer>(B)</answer>\n\n ##Question:'
    ' {mcq_question}\nExplanation: Let us solve this step-by-step, referring to'
    ' authoritative sources as needed. '
)


FITNESS_COT_MCQ = (
    'Instructions: The following are multiple choice questions about fitness'
    ' knowledge. Solve them in a step-by-step fashion, starting by summarizing'
    ' the available information. Output a single option from {mcq_options} as'
    ' the final answer and enclosed by xml tags <answer></answer>.\n\n Here are'
    ' two examples: \n ##Question: A 30-year-old male is looking for a workout'
    ' plan to improve his cardiovascular health. He has no known heart'
    ' conditions and has a sedentary lifestyle. His goal is to increase stamina'
    ' and reduce the risk of heart diseases. Which of the following workout'
    ' plans is most suitable for his goals?\n(A) High-intensity interval'
    ' training (HIIT) 5 days a week\n(B) Moderate-intensity aerobic exercises'
    ' like brisk walking 30 minutes a day, 5 days a week\n(C) Weight training'
    ' focused on major muscle groups 4 days a week\n(D) Yoga and stretching'
    ' exercises twice a week\n(E) Swimming for 60 minutes daily\nExplanation:'
    " Let's solve this step-by-step, referring to authoritative sources as"
    ' needed. For someone with a sedentary lifestyle looking to improve'
    ' cardiovascular health, it is recommended to start with moderate-intensity'
    ' aerobic exercises. This approach is effective in increasing stamina and'
    ' is less likely to cause injury.\nAnswer:'
    ' <answer>(B)</answer>\n\n##Question: What is a common mistake beginners'
    ' make when starting a strength training program?\n(A) Not including enough'
    ' rest days in their routine\n(B) Focusing only on cardio exercises\n(C)'
    ' Lifting weights that are too heavy leading to poor form\n(D) Ignoring'
    ' flexibility and balance training\n(E) Spending too much time on warm-up'
    " exercises\nExplanation: Let's solve this step-by-step, referring to"
    ' authoritative sources as needed. A common mistake for beginners in'
    ' strength training is lifting weights that are too heavy, which can lead'
    ' to poor form and increase the risk of injury.\nAnswer:'
    ' <answer>(C)</answer>\n\n ##Question: {mcq_question}\nExplanation: Let us'
    ' solve this step-by-step, referring to authoritative sources as needed. '
)


SLEEP_TAKE_STEP_BACK_MCQ = (
    'You are a {domain} expert. I want you to solve a multiple-choice question'
    ' in sleep. Here is an example of how to solve a question using an'
    ' abstraction-reasoning approach, starting by recalling relevant principles'
    ' related to the subject of each question. Then apply these principles step'
    ' by step to logically deduce the correct answer. Output a single option'
    ' from {mcq_options} as the final answer and enclosed by xml tags'
    ' <answer></answer>.\n\nHere is an example:\n##Question:\nA 26-year-old'
    ' female presents asking about jet lag. She has no past medical history,'
    ' lives on the East Coast, and travels frequently to the West Coast for'
    ' business. Her career involves planning evening events, and she reports'
    ' significant sleepiness at these events that impairs her ability to'
    ' perform her job. She wants to know how she can adapt to Pacific Standard'
    ' Time (PST) before she travels. What treatment plan will help this patient'
    ' adapt to PST prior to travel?\nOptions:\n(A) Light in evening and later'
    ' bedtime 1 day before traveling\n(B) Light in morning and earlier wake'
    ' time 3 days before traveling\n(C) Light in evening and later bedtime 3'
    ' days before traveling\n(D) Light in morning and earlier wake time 1 month'
    ' before traveling\n(E) Light in evening and later bedtime 1 month before'
    ' traveling\n\n## Principles:\nCircadian Rhythms: The human body operates'
    ' on a circadian rhythm, an internal clock that cycles roughly every 24'
    ' hours. This rhythm is influenced by external cues, especially light.'
    ' Exposure to light can shift the circadian rhythm, making a person feel'
    " more awake or sleepy.\n\nJet Lag: Jet lag occurs when a person's internal"
    ' clock is out of sync with the time zone they are in. This is common when'
    ' traveling across multiple time zones. To adjust to a new time zone, the'
    " body's circadian rhythm needs to be shifted.\n\nLight Therapy: Exposure"
    ' to light at certain times can help shift the circadian rhythm. Exposure'
    ' to light in the morning advances the circadian clock (making one wake up'
    ' earlier), while exposure in the evening delays it (making one stay awake'
    ' later).\n\n## Answer:\nUsing the principles of Circadian Rhythms, Jet'
    ' Lag, and Light Therapy, we can solve the problem as following:\nThe'
    ' patient needs to adapt from the East Coast time to the West Coast time,'
    ' which is 3 hours behind. To do this, she needs to adjust her body clock'
    ' to wake up and go to sleep later according to her current (East Coast)'
    ' time zone, which aligns with the normal waking and sleeping hours in the'
    ' Pacific Standard Time.\n\nLooking at the options:\n- (A) and (C) suggest'
    ' delaying the circadian rhythm (light in the evening and later bedtime),'
    ' which would make her wake up and sleep later according to East Coast'
    ' time. However, this is counterproductive as it would exacerbate the issue'
    ' when she is on the West Coast.\n- (B) Light in the morning and earlier'
    ' wake time 3 days before traveling would advance her circadian rhythm.'
    ' This means she would wake up earlier according to East Coast time, which'
    ' is aligned with waking up at a regular time in PST.\n- (D) and (E)'
    ' propose changes starting 1 month before traveling, which is impractical'
    ' for someone who travels frequently.\n\nTherefore, the correct answer is'
    ' <answer>(B)</answer>. Light in the morning and earlier wake time 3 days'
    ' before traveling. This method would advance her circadian rhythm to'
    ' better match the Pacific Standard Time, helping her to cope with her'
    ' sleepiness during evening events on the West Coast.\n\n ## Question:\n'
    ' {mcq_question}\n\n## Principles:\n'
)


FITNESS_TAKE_STEP_BACK_MCQ = (
    'You are a fitness expert. I want you to solve a multiple-choice question'
    ' in fitness. Here is an example of how to solve a question using an'
    ' abstraction-reasoning approach, starting by recalling relevant principles'
    ' related to the subject of each question. Then apply these principles step'
    ' by step to logically deduce the correct answer. Output a single option'
    ' from {mcq_options} as the final answer and enclosed by xml tags'
    ' <answer></answer>.\n\nHere is an example: \n##Question:\nA 35-year-old'
    ' male is looking to increase muscle mass. He has been working out'
    ' consistently for a year and follows a balanced diet. He wants to know'
    ' which change in his workout routine will be most effective for gaining'
    ' muscle mass. What would you recommend?\nOptions:\n(A) Increase cardio'
    ' exercises and decrease weight lifting\n(B) Focus on high-repetition'
    ' weight lifting with lower weights\n(C) Incorporate high-intensity'
    ' interval training (HIIT) twice a week\n(D) Increase weight lifting with'
    ' heavier weights and lower repetitions\n(E) Maintain the current routine'
    ' without changes\n\n## Principles:\nMuscle Hypertrophy: Muscle growth'
    ' occurs when muscle fibers are damaged and repair themselves, leading to'
    ' an increase in muscle size. This is best achieved through resistance'
    ' training that challenges the muscles.\n\nProgressive Overload: To'
    " continue gaining muscle, it's important to progressively increase the"
    ' demands on the musculoskeletal system. This can be done by lifting'
    ' heavier weights, increasing repetitions, or changing the exercises'
    ' performed.\n\nExercise Variation: Incorporating a variety of exercises'
    ' can help target different muscle groups and prevent plateaus in muscle'
    ' growth.\n\n## Answer:\nUsing the principles of Muscle Hypertrophy,'
    ' Progressive Overload, and Exercise Variation, we can solve the problem'
    ' as following:\nThe individual is already engaged in consistent workouts'
    ' and has a balanced diet, which is fundamental for muscle growth. To'
    ' further enhance muscle mass, the focus should be on increasing the'
    ' intensity of workouts in a way that challenges the muscles more'
    ' significantly.\n\nLooking at the options:\n- (A) focuses on increasing'
    ' cardio, which is less effective for muscle hypertrophy compared to'
    ' resistance training.\n- (B) involves high-repetition lifting with lower'
    ' weights, which is more endurance-focused rather than hypertrophy.\n- (C)'
    ' HIIT can be beneficial for overall fitness but is not the most efficient'
    ' for muscle growth compared to targeted resistance training.\n- (D)'
    ' Increasing weight lifting with heavier weights and lower repetitions is'
    ' aligned with the principles of muscle hypertrophy and progressive'
    ' overload.\n- (E) Maintaining the current routine will not provide the'
    ' necessary stimulus for further muscle growth.\n\nTherefore, the correct'
    ' answer is <answer>(D)</answer>. Increasing weight lifting with heavier'
    ' weights and lower repetitions will effectively promote muscle growth by'
    ' adhering to the principles of muscle hypertrophy and progressive'
    ' overload.\n\n ## Question:\n {mcq_question}\n\n## Principles:\n'
)


def create_prompt_to_generate_mcqs(
    mcq_question: str, mcq_options: dict[str, str], mcq_domain: str
) -> str:
  """Converts a MCQ example to a prompt."""
  if not mcq_question or not mcq_options or not mcq_domain:
    raise ValueError('MCQ example is missing required fields.')
  if mcq_domain not in VALID_HEALTH_DOMAINS:
    raise ValueError(f'MCQ domain {mcq_domain} is not supported.')
  if set(mcq_options) != set(VALID_ANSWER_OPTIONS[: len(mcq_options)]):
    raise ValueError(f'MCQ options are not valid: {mcq_options}')
  instruction = MCQS_EVAL_INSTRUCTION.format(
      domain=mcq_domain, options=', '.join(sorted(mcq_options))
  )
  prompt = instruction + mcq_question
  return prompt
