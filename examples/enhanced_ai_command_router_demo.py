"""
Enhanced AI Command Router Demo - Advanced NLP & Model Selection Examples
Demonstrates the powerful integration of spaCy NLP with AI-powered model recommendations

This example shows:
1. Advanced intent detection using spaCy linguistic analysis
2. AI-powered model recommendations based on detected intent
3. Entity recognition and linguistic feature extraction
4. Smart model selection for specific use cases
5. Educational content detection with advanced markers
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.ai_command_router import EnhancedIntentDetector, AICommandRouter, CommandIntent, IntentResult
from typing import Dict, Any, List

class AICommandRouterDemo:
    """
    Comprehensive demo of the Enhanced AI Command Router capabilities
    Shows real-world examples of advanced NLP processing and model selection
    """
    
    def __init__(self):
        self.intent_detector = EnhancedIntentDetector()
        print("🚀 Enhanced AI Command Router Demo Initialized")
        print("📊 spaCy Available:", hasattr(self.intent_detector, 'nlp') and self.intent_detector.nlp is not None)
        print("🤖 Model Configs Available:", bool(self.intent_detector.model_configs))
        print("=" * 80)

    async def demonstrate_intent_detection(self):
        """Demonstrate advanced intent detection with various message types"""
        print("\n🎯 INTENT DETECTION EXAMPLES")
        print("=" * 50)
        
        # Test messages covering different intents
        test_messages = [
            # Educational Content Detection
            {
                "message": "Can you explain the difference between machine learning and deep learning in detail?",
                "expected": CommandIntent.EDUCATIONAL,
                "description": "Educational query with comparison structure"
            },
            {
                "message": "How to create a comprehensive tutorial on Python virtual environments?",
                "expected": CommandIntent.EDUCATIONAL,
                "description": "Educational request with 'how to' pattern"
            },
            {
                "message": "What are the best practices for REST API design?",
                "expected": CommandIntent.EDUCATIONAL,
                "description": "Educational question about technical topics"
            },
            
            # Document Generation
            {
                "message": "Create a business plan for a tech startup focusing on AI solutions",
                "expected": CommandIntent.GENERATE_DOCUMENT,
                "description": "Document generation with business context"
            },
            {
                "message": "Write me a detailed report on cloud computing trends",
                "expected": CommandIntent.GENERATE_DOCUMENT,
                "description": "Document generation with specific topic"
            },
            
            # Image Generation
            {
                "message": "Create a breathtaking, ultra-realistic 4K image of a vibrant sunset over a majestic mountain range with a crystal-clear lake reflection. The lighting should be dramatic and cinematic, with a rich color palette. The style should be reminiscent of editorial photography with a focus on high detail and a symmetrical composition.",
                "expected": CommandIntent.GENERATE_IMAGE,
                "description": "Image generation with a highly descriptive and stylized prompt"
            },
            {
                "message": "Create an image of a futuristic city with flying cars",
                "expected": CommandIntent.GENERATE_IMAGE,
                "description": "Image generation with creative elements"
            },
            
            # Coding Tasks
            {
                "message": "Write a Python function to implement binary search algorithm",
                "expected": CommandIntent.CODING,
                "description": "Coding request with specific algorithm"
            },
            {
                "message": "Debug this JavaScript code that's not working properly",
                "expected": CommandIntent.CODING,
                "description": "Code debugging request"
            },
            
            # Mathematical Tasks
            {
                "message": "Solve this differential equation: dy/dx = 2x + 3",
                "expected": CommandIntent.MATHEMATICAL,
                "description": "Mathematical problem solving"
            },
            
            # Multilingual Tasks
            {
                "message": "Translate this English text to Chinese and explain cultural context",
                "expected": CommandIntent.MULTILINGUAL,
                "description": "Translation with cultural context"
            },
            
            # Model Switching
            {
                "message": "Switch to a different model please",
                "expected": CommandIntent.SWITCH_MODEL,
                "description": "Model switching request"
            },
            
            # Creative Writing
            {
                "message": "Write a science fiction story about time travel",
                "expected": CommandIntent.CREATIVE,
                "description": "Creative writing request"
            }
        ]
        
        for i, test in enumerate(test_messages, 1):
            print(f"\n📝 Example {i}: {test['description']}")
            print(f"💬 Message: \"{test['message']}\"")
            
            # Detect intent using basic method
            intent, confidence = await self.intent_detector.detect_intent(test['message'])
            
            # Check if detection matches expected
            status = "✅" if intent == test['expected'] else "❌"
            print(f"{status} Detected: {intent.value} (confidence: {confidence:.3f})")
            print(f"🎯 Expected: {test['expected'].value}")
            
            if intent != test['expected']:
                print(f"⚠️  Mismatch detected - may need pattern tuning")

    async def demonstrate_enhanced_analysis(self):
        """Demonstrate enhanced intent analysis with full linguistic features"""
        print("\n🧠 ENHANCED LINGUISTIC ANALYSIS")
        print("=" * 50)
        
        # Create a router instance for enhanced analysis
        router = AICommandRouter(command_handlers=None)  # Mock for demo
        
        complex_messages = [
            {
                "message": "Can you create a comprehensive step-by-step tutorial explaining how machine learning algorithms work, including practical Python examples and mathematical foundations?",
                "description": "Complex educational request with multiple technical terms"
            },
            {
                "message": "I need to debug my neural network implementation in TensorFlow - it's not converging properly during training",
                "description": "Technical coding problem with domain-specific terminology"
            },
            {
                "message": "Generate a detailed business proposal for implementing blockchain technology in supply chain management",
                "description": "Document generation with technical and business context"
            },
            {
                "message": "Write a creative story about AI robots gaining consciousness in a dystopian future society",
                "description": "Creative writing with technical elements"
            }
        ]
        
        for i, test in enumerate(complex_messages, 1):
            print(f"\n🔍 Analysis {i}: {test['description']}")
            print(f"💬 Message: \"{test['message']}\"")
            print("-" * 60)
            
            # Get enhanced analysis
            try:
                result: IntentResult = await router.detect_intent_with_recommendations(test['message'])
                
                print(f"🎯 Intent: {result.intent.value}")
                print(f"📊 Confidence: {result.confidence:.3f}")
                print(f"🧠 Reasoning: {result.reasoning}")
                
                # Show linguistic features
                features = result.linguistic_features
                print(f"📝 Linguistic Features:")
                print(f"   • Tokens: {features.get('tokens', 0)}")
                print(f"   • Sentences: {features.get('sentences', 0)}")
                print(f"   • Technical terms: {features.get('technical_terms', 0)}")
                print(f"   • Complexity score: {features.get('complexity_score', 0):.3f}")
                print(f"   • Has questions: {features.get('questions', False)}")
                print(f"   • Has imperatives: {features.get('imperatives', False)}")
                
                # Show detected entities
                if result.detected_entities:
                    print(f"🏷️  Detected Entities:")
                    for entity in result.detected_entities[:5]:  # Show first 5
                        print(f"   • {entity['text']} ({entity['label']}) - {entity['type']}")
                
                # Show recommended models
                if result.recommended_models:
                    print(f"🤖 Recommended Models:")
                    for model in result.recommended_models:
                        print(f"   • {model}")
                else:
                    print("🤖 No specific model recommendations")
                    
            except Exception as e:
                print(f"❌ Error in enhanced analysis: {e}")

    async def demonstrate_model_recommendations(self):
        """Demonstrate AI-powered model recommendations for different tasks"""
        print("\n🤖 AI-POWERED MODEL RECOMMENDATIONS")
        print("=" * 50)
        
        router = AICommandRouter(command_handlers=None)  # Mock for demo
        
        task_examples = [
            {
                "task": "Educational Content",
                "message": "Explain quantum computing principles with step-by-step examples",
                "expected_models": ["deepseek", "phi-4-reasoning-plus", "llama4-maverick"]
            },
            {
                "task": "Coding Challenge",
                "message": "Implement a sorting algorithm in Python with optimization",
                "expected_models": ["deepcoder", "olympiccoder-32b", "devstral-small"]
            },
            {
                "task": "Mathematical Proof",
                "message": "Prove the Pythagorean theorem using geometric methods",
                "expected_models": ["deepseek-prover-v2", "phi-4-reasoning-plus"]
            },
            {
                "task": "Creative Writing",
                "message": "Write a fantasy novel chapter about magical creatures",
                "expected_models": ["deephermes-3-mistral-24b", "qwerky-72b", "moonlight-16b"]
            },
            {
                "task": "Multilingual Translation",
                "message": "Translate technical documentation from English to Chinese",
                "expected_models": ["qwen3-235b", "glm-z1-32b"]
            },
            {
                "task": "Image Analysis",
                "message": "Analyze this medical scan image for anomalies",
                "expected_models": ["llama-3.2-11b-vision", "qwen2.5-vl-72b"],
                "has_media": True
            }
        ]
        
        for example in task_examples:
            print(f"\n📋 Task: {example['task']}")
            print(f"💬 Request: \"{example['message']}\"")
            print("-" * 40)
            
            try:
                result = await router.detect_intent_with_recommendations(
                    example['message'], 
                    has_attached_media=example.get('has_media', False)
                )
                
                print(f"🎯 Detected Intent: {result.intent.value}")
                print(f"📊 Confidence: {result.confidence:.3f}")
                
                if result.recommended_models:
                    print(f"🤖 AI Recommended Models:")
                    for i, model in enumerate(result.recommended_models, 1):
                        print(f"   {i}. {model}")
                    
                    # Check if recommendations match expected patterns
                    expected = example.get('expected_models', [])
                    if expected:
                        matches = sum(1 for model in result.recommended_models 
                                    if any(exp in model for exp in expected))
                        match_rate = matches / len(expected) if expected else 0
                        print(f"✅ Recommendation accuracy: {match_rate:.1%}")
                else:
                    print("⚠️  No model recommendations available")
                    
            except Exception as e:
                print(f"❌ Error getting recommendations: {e}")

    async def demonstrate_educational_detection(self):
        """Demonstrate advanced educational content detection"""
        print("\n🎓 EDUCATIONAL CONTENT DETECTION")
        print("=" * 50)
        
        educational_examples = [
            # Strong educational markers
            {
                "message": "What is the difference between supervised and unsupervised learning?",
                "markers": ["Question word", "Comparison structure", "Technical terms"]
            },
            {
                "message": "Can you provide a comprehensive tutorial on Docker containerization?",
                "markers": ["Tutorial request", "Comprehensive modifier", "Technical topic"]
            },
            {
                "message": "How to implement RESTful APIs step by step with best practices?",
                "markers": ["How-to pattern", "Step-by-step", "Best practices"]
            },
            {
                "message": "Explain the advantages and disadvantages of microservices architecture",
                "markers": ["Explain verb", "Comparison structure", "Technical architecture"]
            },
            
            # Weak educational markers (should be detected as CHAT)
            {
                "message": "That's interesting, tell me more",
                "markers": ["Conversational", "Low complexity"]
            },
            {
                "message": "Thanks for the help!",
                "markers": ["Gratitude", "Short message"]
            }
        ]
        
        for i, example in enumerate(educational_examples, 1):
            print(f"\n📚 Example {i}")
            print(f"💬 Message: \"{example['message']}\"")
            print(f"🏷️  Expected markers: {', '.join(example['markers'])}")
            
            intent, confidence = await self.intent_detector.detect_intent(example['message'])
            
            if intent == CommandIntent.EDUCATIONAL:
                print(f"✅ Educational content detected (confidence: {confidence:.3f})")
                
                # Show detailed educational scoring if spaCy is available
                if self.intent_detector.nlp:
                    doc = self.intent_detector.nlp(example['message'].lower())
                    edu_score = self.intent_detector._calculate_educational_score(doc)
                    print(f"📊 Educational score breakdown: {edu_score:.3f}")
            else:
                print(f"📝 Detected as: {intent.value} (confidence: {confidence:.3f})")

    async def demonstrate_linguistic_features(self):
        """Demonstrate spaCy linguistic feature extraction"""
        print("\n🔤 LINGUISTIC FEATURE EXTRACTION")
        print("=" * 50)
        
        if not self.intent_detector.nlp:
            print("⚠️  spaCy not available - linguistic features limited")
            return
        
        sample_text = "Can you create a comprehensive tutorial explaining how neural networks work in machine learning? I need detailed explanations with Python code examples and mathematical formulations."
        
        print(f"📝 Analyzing: \"{sample_text}\"")
        print("-" * 60)
        
        doc = self.intent_detector.nlp(sample_text.lower())
        
        # Basic linguistic stats
        print("📊 Basic Linguistic Analysis:")
        print(f"   • Tokens: {len(doc)}")
        print(f"   • Sentences: {len(list(doc.sents))}")
        print(f"   • Named entities: {len(doc.ents)}")
        
        # POS tag analysis
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        print(f"\n🏷️  Part-of-Speech Distribution:")
        for pos, count in sorted(pos_counts.items()):
            if count > 0:
                print(f"   • {pos}: {count}")
        
        # Named entities
        if doc.ents:
            print(f"\n🏷️  Named Entities:")
            for ent in doc.ents:
                print(f"   • {ent.text} ({ent.label_})")
        
        # Dependency relationships
        print(f"\n🔗 Key Dependency Relationships:")
        for token in doc:
            if token.dep_ == 'ROOT':
                print(f"   • Root verb: {token.text} ({token.pos_})")
            elif token.dep_ in ['dobj', 'pobj'] and token.head.pos_ == 'VERB':
                print(f"   • {token.head.text} → {token.text} ({token.dep_})")
        
        # Educational markers
        edu_score = self.intent_detector._calculate_educational_score(doc)
        print(f"\n🎓 Educational Score: {edu_score:.3f}")
        
        # Technical terms
        tech_count = self.intent_detector._count_technical_terms(doc)
        print(f"🔧 Technical Terms Count: {tech_count}")

    async def run_complete_demo(self):
        """Run the complete demo showcasing all features"""
        print("🚀 ENHANCED AI COMMAND ROUTER - COMPLETE DEMO")
        print("=" * 80)
        print("This demo showcases the advanced NLP and AI-powered model selection")
        print("capabilities of the Enhanced AI Command Router system.")
        print("=" * 80)
        
        try:
            await self.demonstrate_intent_detection()
            await self.demonstrate_enhanced_analysis()
            await self.demonstrate_model_recommendations()
            await self.demonstrate_educational_detection()
            await self.demonstrate_linguistic_features()
            
            print("\n" + "=" * 80)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("🎯 The Enhanced AI Command Router provides:")
            print("   • Advanced intent detection using spaCy NLP")
            print("   • AI-powered model recommendations")
            print("   • Sophisticated educational content detection")
            print("   • Comprehensive linguistic feature extraction")
            print("   • Robust fallback mechanisms")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main demo entry point"""
    demo = AICommandRouterDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    # Run the demo
    print("Starting Enhanced AI Command Router Demo...")
    asyncio.run(main())
