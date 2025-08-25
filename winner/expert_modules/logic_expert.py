"""
Logic Expert Module for LogicBench
Specialized expert module for logical reasoning through Wave-based cognition.
"""

import re
import time
import statistics
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, List, Any, Optional, Tuple
from .base_expert import BaseExpertModule, ExpertResponse
from wave_reasoning_engine import WaveReasoningEngine


class LogicExpertModule(BaseExpertModule):
    """
    Expert module for logical reasoning through Wave-based cognition.
    
    Handles different types of logical reasoning:
    - Propositional Logic (modus ponens, modus tollens, etc.)
    - First-order Logic (existential/universal quantification)
    - Non-monotonic Logic (default reasoning with exceptions)
    """
    
    def __init__(self):
        super().__init__("LogicExpert", "logical_reasoning", "1.0")
        
        # Initialize wave reasoning engine for actual reasoning
        self.wave_engine = WaveReasoningEngine("logic_wave_state.pkl")
        
        # Initialize logical rule patterns (now used as scaffolding)
        self.logical_rules = self._initialize_logical_rules()
        
        # Track logical concepts and their wave patterns
        self.logical_concepts = {
            'implication', 'contradiction', 'affirmation', 'negation',
            'universal', 'existential', 'conditional', 'biconditional',
            'conjunction', 'disjunction', 'inference', 'conclusion'
        }
        
    def _define_wave_frequencies(self) -> Dict[str, float]:
        """Define wave frequencies for logical reasoning concepts."""
        return {
            # Core logical operations
            'implication': 2.1,      # If-then relationships
            'negation': 2.8,         # Not operations  
            'conjunction': 3.2,      # And operations
            'disjunction': 3.6,      # Or operations
            'conditional': 4.1,      # Conditional statements
            'biconditional': 4.5,    # If and only if
            
            # Logical rules
            'modus_ponens': 5.1,     # If P then Q, P, therefore Q
            'modus_tollens': 5.4,    # If P then Q, not Q, therefore not P  
            'hypothetical_syllogism': 5.7,  # If P then Q, if Q then R, therefore if P then R
            'disjunctive_syllogism': 6.1,   # P or Q, not P, therefore Q
            'constructive_dilemma': 6.4,    # Complex disjunctive reasoning
            'destructive_dilemma': 6.7,     # Complex destructive reasoning
            
            # First-order logic
            'universal_quantification': 7.2,  # For all x
            'existential_quantification': 7.6, # There exists x
            'universal_instantiation': 8.1,   # From universal to specific
            'existential_instantiation': 8.4,  # From existential to specific
            'existential_generalization': 8.7, # From specific to existential
            
            # Non-monotonic reasoning
            'default_reasoning': 9.2,        # Typical case reasoning
            'exception_handling': 9.6,       # Handling exceptions to rules
            'priority_reasoning': 10.1,      # Resolving conflicting defaults
            
            # Meta-logical concepts
            'contradiction': 11.0,           # Logical contradiction
            'consistency': 11.4,             # Logical consistency
            'validity': 11.8,                # Argument validity
            'soundness': 12.2,               # Argument soundness
        }
    
    def _initialize_logical_rules(self) -> Dict[str, Any]:
        """Initialize logical reasoning rules and patterns."""
        return {
            'modus_ponens': {
                'pattern': r'If\s+(.+?),\s+then\s+(.+?)\.',
                'confidence_boost': 0.2
            },
            'modus_tollens': {
                'pattern': r'If\s+(.+?),\s+then\s+(.+?)\.',
                'confidence_boost': 0.2
            },
            'universal_instantiation': {
                'pattern': r'(?:All|Every|Each)\s+(.+?)\s+(?:are|is|have|has)\s+(.+?)',
                'confidence_boost': 0.18
            }
        }
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Determine if this expert can handle a logical reasoning query."""
        confidence = 0.0
        
        # Check for logical keywords
        logical_keywords = [
            'if', 'then', 'therefore', 'implies', 'entails', 'means', 
            'all', 'some', 'every', 'exists', 'not', 'and', 'or',
            'true', 'false', 'valid', 'invalid', 'consistent', 'contradiction',
            'can we say', 'must', 'always', 'never', 'will', 'won\'t',
            'does', 'doesn\'t', 'is', 'isn\'t', 'are', 'aren\'t'
        ]
        
        query_lower = query.lower()
        keyword_matches = sum(1 for keyword in logical_keywords if keyword in query_lower)
        confidence += min(0.4, keyword_matches * 0.05)
        
        # Check for logical question patterns
        logical_question_patterns = [
            r'can we say.*(?:must|always|true)',
            r'if.*then',
            r'will.*\?',
            r'does.*\?',
            r'at least one.*following.*true',
            r'all.*are',
            r'some.*are'
        ]
        
        for pattern in logical_question_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                confidence += 0.3
                break
        
        # Check context for logical reasoning indicators
        if context:
            if context.get('type') in ['propositional_logic', 'first_order_logic', 'nm_logic']:
                confidence += 0.4
            if context.get('axiom') in self.logical_rules:
                confidence += 0.3
            if 'premises' in context or 'context' in context:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> ExpertResponse:
        """Process a logical reasoning query through Wave-based cognition."""
        start_time = time.time()
        
        # Parse the logical structure
        logical_structure = self._parse_logical_structure(query, context)
        
        # Generate answer based on logical analysis
        answer = self._generate_logical_answer(query, context, logical_structure)
        
        # Calculate confidence
        confidence = self._calculate_logic_confidence(query, context, answer)
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning_explanation(logical_structure, query, context)
        
        # Create wave patterns
        wave_patterns = self._generate_logic_wave_patterns(logical_structure, query)
        
        processing_time = time.time() - start_time
        
        return ExpertResponse(
            confidence=confidence,
            reasoning=reasoning,
            answer=answer,
            wave_patterns=wave_patterns,
            metadata={
                'logical_structure': logical_structure,
                'processing_time': processing_time
            },
            processing_time=processing_time
        )
    
    def _parse_logical_structure(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse the logical structure of the query."""
        structure = {
            'query': query,
            'context': context,
            'logical_type': 'unknown',
            'axiom': 'unknown',
            'premises': [],
            'logical_operators': [],
            'quantifiers': []
        }
        
        # Extract from context if available
        if context:
            structure['logical_type'] = context.get('type', 'unknown')
            structure['axiom'] = context.get('axiom', 'unknown')
            
            # Extract premises from context
            if 'context' in context:
                premises_text = context['context']
                structure['premises'] = self._extract_premises(premises_text)
        
        # Extract logical operators
        operators = {
            'and': r'\b(?:and|&)\b',
            'or': r'\b(?:or|\|)\b',
            'not': r'\b(?:not|¬)\b',
            'implies': r'\b(?:if|then|implies)\b'
        }
        
        for op_name, pattern in operators.items():
            if re.search(pattern, query, re.IGNORECASE):
                structure['logical_operators'].append(op_name)
        
        return structure
    
    def _extract_premises(self, premises_text: str) -> List[str]:
        """Extract individual premises from premise text."""
        sentences = re.split(r'[.!?]+', premises_text)
        premises = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                premises.append(sentence)
        
        return premises
    
    def _generate_logical_answer(self, query: str, context: Dict[str, Any] = None, 
                                logical_structure: Dict[str, Any] = None) -> str:
        """Generate logical answer based on reasoning patterns."""
        
        if not context:
            return self._basic_logical_analysis(query)
        
        # Handle premises-based reasoning first
        if 'premises' in context:
            result = self._handle_premises_reasoning(query, context['premises'])
            if result:
                return result
        
        logic_type = context.get('type', '')
        axiom = context.get('axiom', '')
        
        # Handle different logic types
        if logic_type == 'propositional_logic':
            return self._handle_propositional_logic(query, context, axiom)
        elif logic_type == 'first_order_logic':
            return self._handle_first_order_logic(query, context, axiom)
        elif logic_type == 'nm_logic':
            return self._handle_nm_logic(query, context, axiom)
        else:
            return self._basic_logical_analysis(query)
    
    def _handle_propositional_logic(self, query: str, context: Dict[str, Any], axiom: str) -> str:
        """Handle propositional logic questions."""
        query_lower = query.lower()
        
        # Check for negations in the question
        has_negation = any(neg in query_lower for neg in [
            "won't", "doesn't", "isn't", "will not", "does not", "is not"
        ])
        
        # Simple heuristic based on axiom and negation
        if axiom == 'modus_tollens':
            return "yes" if has_negation else "no"
        elif axiom in ['modus_ponens', 'hypothetical_syllogism', 'disjunctive_syllogism']:
            return "no" if has_negation else "yes"
        else:
            return "yes" if not has_negation else "no"
    
    def _handle_first_order_logic(self, query: str, context: Dict[str, Any], axiom: str) -> str:
        """Handle first-order logic questions."""
        query_lower = query.lower()
        
        # Check for negations
        has_negation = any(neg in query_lower for neg in [
            "not", "doesn't", "isn't", "cannot", "won't"
        ])
        
        if axiom == 'universal_instantiation':
            return "no" if has_negation else "yes"
        elif axiom == 'existential_generalization':
            return "no" if has_negation else "yes"
        else:
            return "yes" if not has_negation else "no"
    
    def _handle_nm_logic(self, query: str, context: Dict[str, Any], axiom: str) -> str:
        """Handle non-monotonic logic questions."""
        query_lower = query.lower()
        
        # Default reasoning patterns
        if 'default_reasoning' in axiom:
            has_positive = any(pos in query_lower for pos in [
                'does', 'is', 'has', 'are', 'usually', 'typically'
            ])
            has_negative = any(neg in query_lower for neg in [
                "doesn't", "isn't", "don't", "aren't", "not"
            ])
            
            if has_positive and not has_negative:
                return "yes"
            else:
                return "no"
        
        # Exception reasoning
        elif 'exception' in axiom:
            has_exactly_one = 'exactly one' in query_lower
            has_negation = any(neg in query_lower for neg in [
                "not", "doesn't", "isn't"
            ])
            
            if has_exactly_one:
                return "yes" if has_negation else "no"
            else:
                return "yes"
        
        else:
            return "no"
    
    def _handle_premises_reasoning(self, query: str, premises: List[str]) -> Optional[str]:
        """Handle reasoning from given premises using wave-based reasoning."""
        query_lower = query.lower()
        
        # Determine reasoning type from premises structure
        rule_type = self._identify_reasoning_type(query_lower, premises)
        
        # Use wave reasoning engine for primary reasoning
        wave_answer, wave_confidence, wave_metadata = self.wave_engine.wave_guided_reasoning(
            query, premises, rule_type
        )
        
        # If wave reasoning is confident, use it (but be more conservative for universal negatives)
        confidence_threshold = 0.75 if rule_type == "universal_negative" else 0.6
        if wave_confidence > confidence_threshold:
            return wave_answer
        
        # Otherwise fall back to rule-based reasoning as backup
        return self._fallback_rule_reasoning(query_lower, premises, rule_type)
    
    def _identify_reasoning_type(self, query: str, premises: List[str]) -> str:
        """Identify the type of logical reasoning needed"""
        text = f"{query} {' '.join(premises)}".lower()
        
        # Check for contradictions first
        if self._detect_contradiction(premises):
            return "contradiction"
        
        # Identify reasoning patterns
        if any(p.lower().startswith('all') for p in premises):
            if 'not' in text:
                return "universal_negative"
            else:
                return "universal_positive"
        elif any('if' in p.lower() and 'then' in p.lower() for p in premises):
            if 'not' in query:
                return "modus_tollens"
            else:
                return "modus_ponens"
        elif any(re.search(r'\bor\b', p.lower()) for p in premises):
            return "disjunctive"
        elif 'if and only if' in text:
            return "biconditional"
        elif any(p.lower().startswith('some') for p in premises):
            return "existential"
        elif len([p for p in premises if 'if' in p.lower() and 'then' in p.lower()]) >= 2:
            return "hypothetical"
        else:
            return "general_reasoning"
    
    def _fallback_rule_reasoning(self, query_lower: str, premises: List[str], rule_type: str) -> Optional[str]:
        """Fallback to rule-based reasoning when wave reasoning is uncertain"""
        
        if rule_type == "contradiction":
            return "contradiction"
        elif rule_type in ["universal_positive", "universal_negative"]:
            return self._try_syllogism_reasoning(query_lower, premises)
        elif rule_type == "modus_ponens":
            return self._try_modus_ponens(query_lower, premises)
        elif rule_type == "modus_tollens":
            return self._try_modus_tollens(query_lower, premises)
        elif rule_type == "disjunctive":
            return self._try_disjunctive_syllogism(query_lower, premises)
        elif rule_type == "hypothetical":
            return self._try_hypothetical_syllogism(query_lower, premises)
        elif rule_type == "biconditional":
            return self._try_biconditional_logic(query_lower, premises)
        elif rule_type == "existential":
            return self._try_existential_quantification(query_lower, premises)
        else:
            return self._try_direct_premise_match(query_lower, premises)
    
    def learn_from_feedback(self, query: str, premises: List[str], expected_answer: str, actual_answer: str, success: bool):
        """Learn from reasoning outcomes to improve future performance"""
        rule_type = self._identify_reasoning_type(query.lower(), premises)
        self.wave_engine.learn_from_outcome(query, premises, rule_type, expected_answer, actual_answer, success)
    
    def get_wave_learning_stats(self) -> Dict[str, Any]:
        """Get current wave learning statistics"""
        return self.wave_engine.get_learning_stats()
    
    def _detect_contradiction(self, premises: List[str]) -> bool:
        """Detect if premises contain contradictions."""
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i != j:
                    p1_lower = premise1.lower().strip()
                    p2_lower = premise2.lower().strip()
                    
                    # Normalize whitespace
                    p1_clean = re.sub(r'\s+', ' ', p1_lower)
                    p2_clean = re.sub(r'\s+', ' ', p2_lower)
                    
                    # Skip if one premise contains "or" - that's disjunctive logic, not contradiction
                    if 'or' in p1_clean or 'or' in p2_clean:
                        continue
                        
                    # Skip if premises are about different subjects (individual vs universal)
                    # E.g., "All X are Y" vs "Z is not Y" is not necessarily a contradiction
                    p1_has_all = 'all' in p1_clean
                    p2_has_all = 'all' in p2_clean
                    if p1_has_all != p2_has_all:
                        continue
                    
                    # Look for direct contradictions like "X is Y" vs "X is not Y"
                    p1_has_negation = any(neg in p1_clean for neg in ['not', "isn't", "doesn't", "cannot", "does not"])
                    p2_has_negation = any(neg in p2_clean for neg in ['not', "isn't", "doesn't", "cannot", "does not"])
                    
                    # Only consider it a contradiction if they have same structure but opposite negation
                    if p1_has_negation != p2_has_negation:
                        # Remove negations and compare
                        p1_no_neg = re.sub(r'\b(not|isn\'t|doesn\'t|cannot|does not)\s+', '', p1_clean)
                        p2_no_neg = re.sub(r'\b(not|isn\'t|doesn\'t|cannot|does not)\s+', '', p2_clean)
                        
                        # Check if they're talking about the same thing
                        p1_no_neg_words = set(re.findall(r'\w+', p1_no_neg))
                        p2_no_neg_words = set(re.findall(r'\w+', p2_no_neg))
                        
                        # High similarity indicates direct contradiction
                        similarity = len(p1_no_neg_words.intersection(p2_no_neg_words)) / max(len(p1_no_neg_words), len(p2_no_neg_words), 1)
                        
                        # Also check for common contradiction patterns
                        is_direct_contradiction = (
                            similarity > 0.8 or
                            # Pattern: "X is always Y" vs "X is not Y" 
                            ('always' in p1_clean and similarity > 0.6) or
                            ('always' in p2_clean and similarity > 0.6) or
                            # Pattern: "X cannot Y" vs "X can Y"
                            ('cannot' in p1_clean and similarity > 0.6) or
                            ('cannot' in p2_clean and similarity > 0.6) or
                            # Pattern: Direct opposites with "does/does not"
                            ('does not' in p1_clean and 'does' in p2_clean and similarity > 0.6) or
                            ('does not' in p2_clean and 'does' in p1_clean and similarity > 0.6) or
                            # Pattern: "X can/cannot" contradictions
                            ('can be' in p1_clean and 'cannot be' in p2_clean) or
                            ('cannot be' in p1_clean and 'can be' in p2_clean) or
                            # Pattern: Flow direction contradictions  
                            ('flows' in p1_clean and 'not flow' in p2_clean) or
                            ('flows' in p2_clean and 'not flow' in p1_clean)
                        )
                        
                        if is_direct_contradiction:
                            return True
        
        return False
    
    def _try_syllogism_reasoning(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply syllogistic reasoning: All X are Y, Z is X, therefore Z is Y."""
        universal_rule = None
        particular_fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Enhanced negative universal statement matching
            # Pattern 1: "All X are not Y"
            all_negative_match = re.search(r'all\s+(\w+)s?\s+are\s+not\s+(\w+)', premise_lower)
            if all_negative_match:
                category = all_negative_match.group(1)
                property_val = all_negative_match.group(2)
                universal_rule = (category, property_val, 'negative')
            # Pattern 2: "No X are Y"  
            elif re.search(r'no\s+(\w+)s?\s+are\s+(\w+)', premise_lower):
                no_match = re.search(r'no\s+(\w+)s?\s+are\s+(\w+)', premise_lower)
                category = no_match.group(1)
                property_val = no_match.group(2)
                universal_rule = (category, property_val, 'negative')
            # Pattern 3: "All X are non-Y" or "All X are un-Y"
            elif re.search(r'all\s+(\w+)s?\s+are\s+(?:non-|un-)?(\w+)', premise_lower):
                non_match = re.search(r'all\s+(\w+)s?\s+are\s+(non-|un-)(\w+)', premise_lower)
                if non_match:
                    category = non_match.group(1)
                    property_val = non_match.group(3)  # Skip the prefix
                    universal_rule = (category, property_val, 'negative')
            else:
                # Look for positive universal statements: "All X are Y" (only if no negation)
                all_positive_match = re.search(r'all\s+(\w+)s?\s+are\s+(\w+)', premise_lower)
                if all_positive_match:
                    category = all_positive_match.group(1)
                    property_val = all_positive_match.group(2)
                    universal_rule = (category, property_val, 'positive')
            
            # Look for particular statements: "Z is X"
            is_match = re.search(r'(\w+)\s+is\s+a?\s*(\w+)', premise_lower)
            if is_match:
                individual = is_match.group(1)
                category = is_match.group(2)
                particular_fact = (individual, category)
        
        # Apply syllogism if we have both parts
        if universal_rule and particular_fact:
            category, property_val, rule_type = universal_rule
            individual, individual_category = particular_fact
            
            # Handle singular/plural matching
            category_singular = category.rstrip('s') if category.endswith('s') else category
            individual_category_singular = individual_category.rstrip('s') if individual_category.endswith('s') else individual_category
            
            # Enhanced category matching with more flexibility
            category_match = (individual_category_singular == category_singular or 
                            individual_category == category or
                            category_singular in individual_category_singular or
                            individual_category_singular in category_singular)
            
            if category_match:
                # Enhanced property and individual matching
                property_singular = property_val.rstrip('s') if property_val.endswith('s') else property_val
                property_variations = [property_val, property_singular, property_val + 's']
                
                query_has_individual = individual in query_lower
                query_has_property = any(prop in query_lower for prop in property_variations)
                
                # Additional check for word boundaries and variations
                query_words = set(re.findall(r'\w+', query_lower))
                individual_match = individual in query_words or individual.lower() in query_words
                property_word_match = any(prop in query_words for prop in property_variations)
                
                if (query_has_individual or individual_match) and (query_has_property or property_word_match):
                    # Enhanced logic for negative rules
                    if rule_type == 'negative':
                        # Extract just the question part (usually after the last period or question mark)
                        question_parts = [part.strip() for part in re.split(r'[.!?]+', query_lower) if part.strip()]
                        actual_question = question_parts[-1] if question_parts else query_lower
                        
                        # Check if the actual question (not premises) is asking about negation
                        query_has_negation = any(neg in actual_question for neg in ['not', "isn't", "doesn't", 'no '])
                        # If query has negation and we have negative rule → "yes"
                        # If query has no negation and we have negative rule → "no"  
                        return "yes" if query_has_negation else "no"
                    else:
                        # Positive rule: "All X are Y" → "yes"
                        return "yes"
        
        return None
    
    def _try_modus_ponens(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply modus ponens: If P then Q, P, therefore Q."""
        conditional = None
        fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for conditionals: "If P then Q"
            if_match = re.search(r'if\s+(.+?)\s+then\s+(.+)', premise_lower)
            if if_match:
                condition = if_match.group(1).strip()
                consequence = if_match.group(2).strip()
                conditional = (condition, consequence)
            else:
                # Look for simple facts
                if not any(word in premise_lower for word in ['if', 'then', 'all']) and len(premise.strip()) > 0:
                    fact = premise_lower.strip()
        
        # Apply modus ponens
        if conditional and fact:
            condition, consequence = conditional
            
            # Check if fact matches condition
            if condition in fact or fact in condition:
                # Check if query asks about consequence
                consequence_words = set(re.findall(r'\w+', consequence))
                query_words = set(re.findall(r'\w+', query_lower))
                if consequence_words.intersection(query_words):
                    return "yes"
        
        return None
    
    def _try_modus_tollens(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply modus tollens: If P then Q, not Q, therefore not P."""
        conditional = None
        negative_fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for conditionals: "If P then Q"
            if_match = re.search(r'if\s+(.+?)\s+then\s+(.+)', premise_lower)
            if if_match:
                condition = if_match.group(1).strip()
                consequence = if_match.group(2).strip()
                conditional = (condition, consequence)
            else:
                # Look for negated facts: "X is not Y" or "It is not the case that X"
                if any(neg in premise_lower for neg in ['not', "isn't", "doesn't", 'no']):
                    negative_fact = premise_lower.strip()
        
        # Apply modus tollens
        if conditional and negative_fact:
            condition, consequence = conditional
            
            # Check if negative fact contradicts the consequence
            consequence_words = set(re.findall(r'\w+', consequence))
            negative_fact_words = set(re.findall(r'\w+', negative_fact))
            
            # Remove negation words for comparison
            negative_fact_clean = re.sub(r'\b(not|isn\'t|doesn\'t|no)\s*', '', negative_fact).strip()
            negative_fact_clean_words = set(re.findall(r'\w+', negative_fact_clean))
            
            # If the negative fact contradicts the consequence, we can conclude not P
            if consequence_words.intersection(negative_fact_clean_words):
                # Check if query asks about negation of the condition
                condition_words = set(re.findall(r'\w+', condition))
                query_words = set(re.findall(r'\w+', query_lower))
                
                if condition_words.intersection(query_words):
                    # If query has negation, answer yes; if no negation, answer no
                    query_has_negation = any(neg in query_lower for neg in ['not', "isn't", "doesn't"])
                    return "yes" if query_has_negation else "no"
        
        return None
    
    def _try_disjunctive_syllogism(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply disjunctive syllogism: P or Q, not P, therefore Q."""
        disjunction = None
        negation = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for disjunction: "P or Q" or "Either P or Q"
            or_match = re.search(r'(?:either\s+)?(.+?)\s+or\s+(.+)', premise_lower)
            if or_match:
                option1 = or_match.group(1).strip()
                option2 = or_match.group(2).strip()
                disjunction = (option1, option2)
            else:
                # Look for negations
                if any(neg in premise_lower for neg in ['not', "isn't", "doesn't", 'no']):
                    negation = premise_lower.strip()
        
        # Apply disjunctive syllogism
        if disjunction and negation:
            option1, option2 = disjunction
            
            # Remove negation words for comparison
            negation_clean = re.sub(r'\b(not|isn\'t|doesn\'t|no)\s*', '', negation).strip()
            negation_words = set(re.findall(r'\w+', negation_clean))
            
            option1_words = set(re.findall(r'\w+', option1))
            option2_words = set(re.findall(r'\w+', option2))
            
            # If negation contradicts option1, conclude option2
            if option1_words.intersection(negation_words):
                query_words = set(re.findall(r'\w+', query_lower))
                if option2_words.intersection(query_words):
                    return "yes"
            
            # If negation contradicts option2, conclude option1  
            elif option2_words.intersection(negation_words):
                query_words = set(re.findall(r'\w+', query_lower))
                if option1_words.intersection(query_words):
                    return "yes"
        
        return None
    
    def _try_hypothetical_syllogism(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply hypothetical syllogism: If A then B, If B then C, therefore If A then C."""
        conditionals = []
        
        # Extract all conditionals
        for premise in premises:
            premise_lower = premise.lower()
            if_match = re.search(r'if\s+(.+?)\s+then\s+(.+)', premise_lower)
            if if_match:
                condition = if_match.group(1).strip()
                consequence = if_match.group(2).strip()
                conditionals.append((condition, consequence))
        
        # Look for chain: If A then B, If B then C
        if len(conditionals) >= 2:
            for i, (cond1, cons1) in enumerate(conditionals):
                for j, (cond2, cons2) in enumerate(conditionals):
                    if i != j:
                        # Check if consequence of first matches condition of second
                        cons1_words = set(re.findall(r'\w+', cons1))
                        cond2_words = set(re.findall(r'\w+', cond2))
                        
                        if cons1_words.intersection(cond2_words):
                            # We have a chain: cond1 -> cons1 -> cons2
                            # Check if query asks about cond1 -> cons2
                            query_words = set(re.findall(r'\w+', query_lower))
                            cond1_words = set(re.findall(r'\w+', cond1))
                            cons2_words = set(re.findall(r'\w+', cons2))
                            
                            # Query should ask "If cond1 then cons2?" or similar
                            if ('if' in query_lower and 'then' in query_lower and 
                                cond1_words.intersection(query_words) and 
                                cons2_words.intersection(query_words)):
                                return "yes"
        
        return None
    
    def _try_disjunctive_syllogism(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply disjunctive syllogism: Either A or B, not A, therefore B."""
        disjunction = None
        negated_fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for "Either A or B" or "A or B" 
            either_or_match = re.search(r'(?:either\s+)?(.+?)\s+(?:is\s+)?(?:or\s+)(.+)', premise_lower)
            if either_or_match:
                option_a = either_or_match.group(1).strip()
                option_b = either_or_match.group(2).strip()
                disjunction = (option_a, option_b)
            
            # Look for negated facts: "The X is not Y" or "X is not Y"
            elif 'not' in premise_lower:
                # Remove "the" and extract the negated fact
                cleaned = re.sub(r'\bthe\b', '', premise_lower).strip()
                not_match = re.search(r'(.+?)\s+(?:is\s+)?not\s+(.+)', cleaned)
                if not_match:
                    subject = not_match.group(1).strip()
                    negated_property = not_match.group(2).strip()
                    negated_fact = (subject, negated_property)
        
        # Apply disjunctive elimination
        if disjunction and negated_fact:
            option_a, option_b = disjunction
            negated_subject, negated_property = negated_fact
            
            # Extract words for matching
            option_a_words = set(re.findall(r'\w+', option_a))
            option_b_words = set(re.findall(r'\w+', option_b))
            negated_words = set(re.findall(r'\w+', f"{negated_subject} {negated_property}"))
            query_words = set(re.findall(r'\w+', query_lower))
            
            # If negated fact matches option A, then option B must be true
            if option_a_words.intersection(negated_words):
                if option_b_words.intersection(query_words):
                    return "yes"
            
            # If negated fact matches option B, then option A must be true  
            elif option_b_words.intersection(negated_words):
                if option_a_words.intersection(query_words):
                    return "yes"
        
        return None
    
    def _try_biconditional_logic(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply biconditional logic: P if and only if Q (P ↔ Q)."""
        biconditional = None
        fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for "if and only if" statements
            iff_match = re.search(r'(.+?)\s+if\s+and\s+only\s+if\s+(.+)', premise_lower)
            if iff_match:
                left = iff_match.group(1).strip()
                right = iff_match.group(2).strip()
                biconditional = (left, right)
            else:
                # Look for simple facts
                if not any(word in premise_lower for word in ['if', 'only', 'then']) and len(premise.strip()) > 0:
                    fact = premise_lower.strip()
        
        # Apply biconditional reasoning with enhanced wave-guided matching
        if biconditional and fact:
            left, right = biconditional
            
            left_words = set(re.findall(r'\w+', left))
            right_words = set(re.findall(r'\w+', right))
            fact_words = set(re.findall(r'\w+', fact))
            query_words = set(re.findall(r'\w+', query_lower))
            
            # Enhanced matching with multiple strategies
            left_overlap = len(left_words.intersection(fact_words)) / max(len(left_words), 1)
            right_overlap = len(right_words.intersection(fact_words)) / max(len(right_words), 1)
            
            # Strategy 1: More aggressive threshold matching
            if left_overlap >= 0.25:
                if right_words.intersection(query_words):
                    return "yes"
            elif right_overlap >= 0.25:
                if left_words.intersection(query_words):
                    return "yes"
            
            # Strategy 2: Key concept matching - focus on meaningful words
            left_core = {word for word in left_words if len(word) > 3 and word not in {'that', 'with', 'from', 'they', 'when', 'only', 'this'}}
            right_core = {word for word in right_words if len(word) > 3 and word not in {'that', 'with', 'from', 'they', 'when', 'only', 'this'}}
            fact_core = {word for word in fact_words if len(word) > 3 and word not in {'that', 'with', 'from', 'they', 'when', 'only', 'this'}}
            
            if left_core and fact_core and left_core.intersection(fact_core):
                if right_core.intersection(query_words) or right_words.intersection(query_words):
                    return "yes"
            elif right_core and fact_core and right_core.intersection(fact_core):
                if left_core.intersection(query_words) or left_words.intersection(query_words):
                    return "yes"
            
            # Strategy 3: Partial matching for word variations
            for left_word in left_words:
                for fact_word in fact_words:
                    if (left_word in fact_word or fact_word in left_word) and len(left_word) > 2:
                        if any(qw in right_words or any(rw in qw or qw in rw for rw in right_words if len(rw) > 2) for qw in query_words):
                            return "yes"
            
            for right_word in right_words:
                for fact_word in fact_words:
                    if (right_word in fact_word or fact_word in right_word) and len(right_word) > 2:
                        if any(qw in left_words or any(lw in qw or qw in lw for lw in left_words if len(lw) > 2) for qw in query_words):
                            return "yes"
        
        return None
    
    def _try_existential_quantification(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Handle existential quantification: Some X are Y reasoning."""
        existential_statements = []
        particular_fact = None
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Look for "Some X are Y" statements
            some_match = re.search(r'some\s+(\w+)s?\s+are\s+(\w+)', premise_lower)
            if some_match:
                category = some_match.group(1)
                property_val = some_match.group(2)
                existential_statements.append((category, property_val))
            
            # Look for particular facts: "Z is X"  
            is_match = re.search(r'(\w+)\s+is\s+a?\s*(\w+)', premise_lower)
            if is_match:
                individual = is_match.group(1)
                category = is_match.group(2)
                particular_fact = (individual, category)
        
        # Apply existential reasoning
        for category, property_val in existential_statements:
            if particular_fact:
                individual, individual_category = particular_fact
                
                # Handle singular/plural matching
                category_singular = category.rstrip('s') if category.endswith('s') else category
                individual_category_singular = individual_category.rstrip('s') if individual_category.endswith('s') else individual_category
                
                if individual_category_singular == category_singular or individual_category == category:
                    query_words = set(re.findall(r'\w+', query_lower))
                    individual_words = {individual}
                    property_words = {property_val, property_val.rstrip('s')}
                    
                    # If query asks about the individual having the property
                    if (individual_words.intersection(query_words) and 
                        property_words.intersection(query_words)):
                        return "possible"  # Existential gives possibility, not certainty
        
        # Check if query asks about existence directly
        might_match = re.search(r'might\s+(\w+)\s+be\s+(\w+)', query_lower)
        if might_match:
            individual = might_match.group(1)
            property_val = might_match.group(2)
            
            for category, exist_property in existential_statements:
                if particular_fact:
                    indiv, indiv_cat = particular_fact
                    if (indiv.lower() == individual.lower() and 
                        exist_property.lower() == property_val.lower()):
                        return "yes"
        
        return None
    
    def _try_logical_equivalences(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Apply De Morgan's Laws and Contrapositive reasoning."""
        
        for premise in premises:
            premise_lower = premise.lower()
            
            # Contrapositive: If A then B ≡ If not B then not A
            if_match = re.search(r'if\s+(.+?)\s+then\s+(.+)', premise_lower)
            if if_match:
                condition = if_match.group(1).strip()
                consequence = if_match.group(2).strip()
                
                # Check if query asks about contrapositive
                query_words = set(re.findall(r'\w+', query_lower))
                condition_words = set(re.findall(r'\w+', condition))
                consequence_words = set(re.findall(r'\w+', consequence))
                
                # Query: \"If not consequence then not condition?\"
                if ('if' in query_lower and 'not' in query_lower and
                    consequence_words.intersection(query_words) and
                    condition_words.intersection(query_words)):
                    return "yes"
            
            # De Morgan's Law: not(A and B) ≡ (not A) or (not B)
            not_and_match = re.search(r'not\s*\\((.+?)\s+and\s+(.+?)\\)', premise_lower)
            if not_and_match:
                part_a = not_and_match.group(1).strip()
                part_b = not_and_match.group(2).strip()
                
                # Check if query asks about equivalent form
                query_words = set(re.findall(r'\w+', query_lower))
                a_words = set(re.findall(r'\w+', part_a))
                b_words = set(re.findall(r'\w+', part_b))
                
                # Query asks about \"not A or not B\"
                if ('not' in query_lower and 'or' in query_lower and
                    (a_words.intersection(query_words) or b_words.intersection(query_words))):
                    return "yes"
            
            # De Morgan's Law: not(A or B) ≡ (not A) and (not B)  
            not_or_match = re.search(r'not\s*\\((.+?)\s+or\s+(.+?)\\)', premise_lower)
            if not_or_match:
                part_a = not_or_match.group(1).strip()
                part_b = not_or_match.group(2).strip()
                
                query_words = set(re.findall(r'\w+', query_lower))
                a_words = set(re.findall(r'\w+', part_a))
                b_words = set(re.findall(r'\w+', part_b))
                
                # Query asks about \"not A and not B\"
                if ('not' in query_lower and 'and' in query_lower and
                    a_words.intersection(query_words) and b_words.intersection(query_words)):
                    return "yes"
        
        return None
    
    def _try_direct_premise_match(self, query_lower: str, premises: List[str]) -> Optional[str]:
        """Try direct matching with premises."""
        for premise in premises:
            premise_lower = premise.lower()
            
            # Extract key words from query and premise
            query_words = set(re.findall(r'\w+', query_lower))
            premise_words = set(re.findall(r'\w+', premise_lower))
            
            # If significant overlap, might be directly answerable
            overlap = query_words.intersection(premise_words)
            if len(overlap) >= 2:
                # Check for negation in query
                has_negation = any(neg in query_lower for neg in ['not', "isn't", "doesn't"])
                
                # Check for negation in premise
                premise_has_negation = any(neg in premise_lower for neg in ['not', "isn't", "doesn't"])
                
                # If both have negation or neither have negation, return yes
                # If one has negation and other doesn't, return no
                if has_negation == premise_has_negation:
                    return "yes"
                else:
                    return "no"
        
        return None
    
    def _basic_logical_analysis(self, query: str) -> str:
        """Basic logical analysis for queries without context."""
        query_lower = query.lower()
        
        # Simple yes/no heuristics
        if any(pattern in query_lower for pattern in ['if', 'then', 'implies']):
            return "yes"
        elif any(pattern in query_lower for pattern in ["won't", "doesn't", "not"]):
            return "no"
        else:
            return "yes"
    
    def _calculate_logic_confidence(self, query: str, context: Dict[str, Any] = None, answer: str = "") -> float:
        """Calculate confidence in logical reasoning."""
        base_confidence = 0.6
        
        # Boost confidence for recognized patterns
        if context:
            if context.get('type') in ['propositional_logic', 'first_order_logic', 'nm_logic']:
                base_confidence += 0.2
            if context.get('axiom') in self.logical_rules:
                base_confidence += 0.1
        
        # Boost confidence for clear logical structure
        if any(word in query.lower() for word in ['if', 'then', 'all', 'some']):
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _generate_reasoning_explanation(self, logical_structure: Dict[str, Any], 
                                      query: str, context: Dict[str, Any] = None) -> str:
        """Generate reasoning explanation."""
        explanation_parts = []
        
        # Explain the logical structure
        axiom = logical_structure.get('axiom', 'unknown')
        if axiom != 'unknown':
            explanation_parts.append(f"Identified logical rule: {axiom}")
        
        logical_type = logical_structure.get('logical_type', 'unknown')
        if logical_type != 'unknown':
            explanation_parts.append(f"Logic type: {logical_type}")
        
        # Explain operators found
        if logical_structure.get('logical_operators'):
            ops = logical_structure['logical_operators']
            explanation_parts.append(f"Logical operators: {', '.join(ops)}")
        
        explanation_parts.append("Applied wave-based logical reasoning")
        
        return " | ".join(explanation_parts)
    
    def _generate_logic_wave_patterns(self, logical_structure: Dict[str, Any], query: str) -> Dict[str, float]:
        """Generate wave patterns for logical concepts."""
        patterns = {}
        
        # Add patterns for logical type
        logical_type = logical_structure.get('logical_type', 'unknown')
        if logical_type in self.wave_frequencies:
            patterns[logical_type] = 0.8
        
        # Add patterns for axiom
        axiom = logical_structure.get('axiom', 'unknown')
        if axiom in self.wave_frequencies:
            patterns[axiom] = 0.9
        
        # Add patterns for operators
        for operator in logical_structure.get('logical_operators', []):
            if operator in self.wave_frequencies:
                patterns[operator] = 0.6
        
        return patterns