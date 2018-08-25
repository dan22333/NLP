relations_list = ['analogy', 'Analogy', 'antithesis', 'Contrast', 'attribution', 
		'attribution-n', 'background', 'cause', 'Cause-Result', 'circumstance',
		'comparison', 'Comparison', 'comment', 'Comment-Topic', 'concession',
		'conclusion', 'Conclusion','condition', 'consequence-s', 'consequence-n', 'Consequence',
		'contingency', 'Contrast', 'definition', 'Disjunction', 'elaboration-additional', 
		'elaboration-set-member', 'elaboration-part-whole', 'elaboration-process-step',
		'elaboration-object-attribute','elaboration-general-specific', 'enablement',
		'evaluation-s', 'evaluation-n', 'Evaluation', 'evidence', 'example',
		'explanation-argumentative', 'hypothetical', 'interpretation-s', 
		'interpretation-n', 'Interpretation', 'Inverted-Sequence',
		'List', 'manner', 'means', 'otherwise', 'Otherwise', 'preference',
		'problem-solution-s', 'problem-solution-n', 'Problem-Solution',
		'Proportion', 'purpose', 'question-answer-s', 'question-answer-n', 'Question-Answer',
		'reason', 'Reason', 'restatement', 'result', 'Cause-Result', 'rhetorical-question',
		'Same-Unit', 'Sequence', 'statement-response-s', 'statement-response-n', 'Statement-Response',
		'summary-s summary-n', 'temporal-before', 'temporal-same-time', 'temporal-same-time',
		'Temporal-Same-Time', 'temporal-after', 'TextualOrganization', 'Topic-Comment',
		'topic-drift', 'Topic-Drift', 'topic-shift', 'Topic-Shift']


multi_nuclear_rels = { 
	'Analogy' : 1, 'Cause-Result' : 1, 'Comparison': 1, 'Comment-Topic': 1, 'Conclusion' : 1, 'Consequence' : 1,
	'Contrast' : 1, 'Disjunction' : 1, 'Evaluation' : 1, 'Interpretation' : 1, 
	'Inverted-Sequence' : 1, 'List' : 1, 'Otherwise' : 1 , 'Problem-Solution': 1, 
	'Proportion' : 1, 'Question-Answer' : 1, 'Reason' : 1, 'Topic-Drift': 1, 
	'Cause-Result' : 1, 'Same-Unit' : 1, 'Sequence' : 1, 'Statement-Response' : 1, 
	'Temporal-Same-Time' : 1, 'TextualOrganization' : 1, 'Topic-Comment' : 1,
	'Topic-Drift' : 1, 'Topic-Shift': 1,  
	}

mono_nuclearity_nuc_rels = { 
	'cause' : 1, 'consequence-n' : 1, 'evaluation-n' : 1, 'interpretation-n' : 1, 
	'problem-solution-n' : 1, 'question-answer-n' : 1, 'statement-response-n': 1, 'summary-n': 1, 'temporal-before' : 1, 
	'temporal-same-time' : 1, 'temporal-same-time' : 1, 'temporal-after': 1
	}

mono_nuclearity_sate_rels = { 
	'analogy' : 1, 'antithesis' : 1, 'attribution' : 1, 'problem-solution-n': 1, 
	'attribution-n' : 1, 'background' : 1,   'circumstance' : 1, 'comparison' : 1, 
	'comment': 1, 'concession' : 1, 'conclusion' :1, 'condition' : 1, 
	'consequence-s' : 1, 'contingency' : 1,  'definition' : 1, 
	'elaboration-additional' : 1, 'elaboration-set-member' : 1, 
	'elaboration-part-whole' : 1, 'elaboration-process-step' : 1,
	'elaboration-object-attribute' : 1,'elaboration-general-specific' : 1, 'enablement' : 1,
	'evaluation-s' : 1, 'evidence' : 1, 'example' : 1,
	'explanation-argumentative' : 1, 'hypothetical' : 1, 'interpretation-s' : 1, 
	'manner' : 1, 'means' : 1, 'otherwise' : 1,  'preference' : 1, 'problem-solution-s' : 1,
	'purpose' : 1, 'question-answer-s' : 1,
	'reason' : 1,  'restatement' : 1, 'result' : 1, 'rhetorical-question' : 1,
	'statement-response-s' : 1, 'summary-s': 1,  'topic-shift': 1 
	}

def is_mono_nuclearity_satellite_relation(rel):
	return mono_nuclearity_sate_rels.get(rel)

def is_mono_nuclearity_nucleus_relation(rel):
	return mono_nuclearity_nuc_rels.get(rel)

def is_multi_nuclear_relation(rel):
	return multi_nuclear_rels.get(rel)

cluster_rels_list = ["ATTRIBUTION", "BACKGROUND", "CAUSE", "COMPARISON", 
	"CONDITION", "CONTRAST", "ELABORATION", "ENABLEMENT", "TOPICCOMMENT", 
	"EVALUATION", "EXPLANATION", "JOINT", "MANNERMEANS", "SUMMARY", "TEMPORAL",
	"TOPICCHANGE", "SPAN", "SAME-UNIT", "TEXTUALORGANIZATION"]

action_to_ind_map = {
	'SHIFT': 0, 'REDUCE-ATTRIBUTION-RIGHT': 1, 'REDUCE-BACKGROUND-RIGHT': 2, 'REDUCE-CAUSE-RIGHT': 3, 
	'REDUCE-COMPARISON-RIGHT': 4, 'REDUCE-CONDITION-RIGHT': 5, 'REDUCE-CONTRAST-RIGHT': 6, 
	'REDUCE-ELABORATION-RIGHT': 7, 'REDUCE-ENABLEMENT-RIGHT': 8, 
	'REDUCE-TOPICCOMMENT-RIGHT': 9, 'REDUCE-EVALUATION-RIGHT': 10, 'REDUCE-EXPLANATION-RIGHT': 11, 
	'REDUCE-JOINT-RIGHT': 12, 'REDUCE-MANNERMEANS-RIGHT': 13, 'REDUCE-SUMMARY-RIGHT': 14, 
	'REDUCE-TEMPORAL-RIGHT': 15, 'REDUCE-TOPICCHANGE-RIGHT': 16, 'REDUCE-SPAN-RIGHT': 17, 
	'REDUCE-SAME-UNIT-RIGHT': 18, 'REDUCE-TEXTUALORGANIZATION-RIGHT': 19, 
	'REDUCE-ATTRIBUTION-LEFT': 20, 'REDUCE-BACKGROUND-LEFT': 21, 'REDUCE-CAUSE-LEFT': 22, 
	'REDUCE-COMPARISON-LEFT': 23, 'REDUCE-CONDITION-LEFT': 24, 'REDUCE-CONTRAST-LEFT': 25, 
	'REDUCE-ELABORATION-LEFT': 26, 'REDUCE-ENABLEMENT-LEFT': 27, 'REDUCE-TOPICCOMMENT-LEFT': 28,
	'REDUCE-EVALUATION-LEFT': 29, 'REDUCE-EXPLANATION-LEFT': 30, 
	'REDUCE-JOINT-LEFT': 31, 'REDUCE-MANNERMEANS-LEFT': 32, 'REDUCE-SUMMARY-LEFT': 34, 
	'REDUCE-TEMPORAL-LEFT': 35, 'REDUCE-TOPICCHANGE-LEFT': 36, 'REDUCE-SPAN-LEFT': 37, 
	'REDUCE-SAME-UNIT-LEFT': 38, 'REDUCE-TEXTUALORGANIZATION-LEFT': 39
	}

ind_to_action_map = ['SHIFT', 'REDUCE-ATTRIBUTION-RIGHT', 'REDUCE-BACKGROUND-RIGHT', 'REDUCE-CAUSE-RIGHT', 
	'REDUCE-COMPARISON-RIGHT', 'REDUCE-CONDITION-RIGHT', 'REDUCE-CONTRAST-RIGHT', 
	'REDUCE-ELABORATION-RIGHT', 'REDUCE-ENABLEMENT-RIGHT', 
	'REDUCE-TOPICCOMMENT-RIGHT', 'REDUCE-EVALUATION-RIGHT', 'REDUCE-EXPLANATION-RIGHT', 
	'REDUCE-JOINT-RIGHT', 'REDUCE-MANNERMEANS-RIGHT', 'REDUCE-SUMMARY-RIGHT', 
	'REDUCE-TEMPORAL-RIGHT', 'REDUCE-TOPICCHANGE-RIGHT', 'REDUCE-SPAN-RIGHT', 
	'REDUCE-SAME-UNIT-RIGHT', 'REDUCE-TEXTUALORGANIZATION-RIGHT', 
	'REDUCE-ATTRIBUTION-LEFT', 'REDUCE-BACKGROUND-LEFT' 'REDUCE-CAUSE-LEFT', 
	'REDUCE-COMPARISON-LEFT', 'REDUCE-CONDITION-LEFT', 'REDUCE-CONTRAST-LEFT', 
	'REDUCE-ELABORATION-LEFT', 'REDUCE-ENABLEMENT-LEFT', 'REDUCE-TOPICCOMMENT-LEFT',
	'REDUCE-EVALUATION-LEFT', 'REDUCE-EXPLANATION-LEFT', 
	'REDUCE-JOINT-LEFT', 'REDUCE-MANNERMEANS-LEFT', 'REDUCE-SUMMARY-LEFT', 
	'REDUCE-TEMPORAL-LEFT', 'REDUCE-TOPICCHANGE-LEFT', 'REDUCE-SPAN-LEFT', 
	'REDUCE-SAME-UNIT-LEFT', 'REDUCE-TEXTUALORGANIZATION-LEFT']



  
