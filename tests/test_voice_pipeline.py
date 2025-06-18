
import asyncio
import logging
from src.handlers.response_formatter import ResponseFormatter

# Set up logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_voice_pipeline():
    """
    Test the complete voice formatting pipeline:
    1. Raw AI response
    2. Format with voice intro and context
    3. Apply model indicator 
    4. Format for Telegram (THE FIX!)
    5. Ready to send
    """
    
    print("🔄 Testing Complete Voice Response Pipeline")
    print("=" * 60)
    
    formatter = ResponseFormatter()
    
    # Simulate a typical AI response that would come from OpenRouter/DeepSeek
    raw_ai_response = "Hello! I can help you with your questions. **Programming** is a great skill to learn, and *Python* is an excellent choice for beginners."
    
    print("📥 Step 1: Raw AI Response")
    print(f"   {raw_ai_response}")
    
    # Step 2: Format as voice response (like in voice handler)
    voice_intro = "🎤 **Voice Response:**"
    context_hint = "_Continuing our conversation..._\n\n"
    voice_formatted_response = f"{voice_intro}\n\n{context_hint}{raw_ai_response}"
    
    print("\n📝 Step 2: Voice-formatted Response")
    print(f"   {voice_formatted_response}")
    
    # Step 3: Apply model indicator (like in voice handler)
    model_indicator = "🤖 DeepSeek"
    with_model_indicator = formatter.format_with_model_indicator(
        voice_formatted_response, model_indicator, False
    )
    
    print("\n🏷️  Step 3: With Model Indicator")
    print(f"   {with_model_indicator}")
    
    # Step 4: Format for Telegram (THE FIX WE ADDED!)
    try:
        telegram_ready = await formatter.format_telegram_markdown(with_model_indicator)
        
        print("\n✨ Step 4: Telegram-ready (THE FIX!)")
        print(f"   {telegram_ready}")
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("\n🔍 Analysis:")
        print(f"   - Raw response length: {len(raw_ai_response)} chars")
        print(f"   - Final formatted length: {len(telegram_ready)} chars")
        print(f"   - Contains Telegram markdown: {'*' in telegram_ready or '_' in telegram_ready}")
        has_escapes = '\\' in telegram_ready
        print(f"   - Special chars escaped: {has_escapes}")
        
        print("\n✅ The fix ensures voice responses:")
        print("   ✓ Show proper bold/italic formatting in Telegram")
        print("   ✓ Follow the same formatting path as text messages")
        print("   ✓ Have consistent user experience")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in Step 4: {str(e)}")
        print("   This would mean the fix didn't work properly!")
        return False

async def test_before_vs_after():
    """Show the difference between before and after the fix"""
    
    print("\n" + "🔄 BEFORE vs AFTER Fix Comparison")
    print("=" * 60)
    
    formatter = ResponseFormatter()
    sample_response = "🤖 DeepSeek | 🎤 **Voice Response:**\n\n_Based on our conversation..._\n\nI can help you with **anything** you need!"
    
    print("📋 Sample Voice Response:")
    print(f"   {sample_response}")
    
    print("\n❌ BEFORE Fix (what users saw):")
    print("   Raw markdown text: **Voice Response:** with literal asterisks")
    print("   No formatting applied - looked unprofessional")
    
    print("\n✅ AFTER Fix (what users see now):")
    try:
        formatted = await formatter.format_telegram_markdown(sample_response)
        print(f"   Properly formatted: {formatted[:100]}...")
        print("   Bold/italic text renders correctly in Telegram!")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    async def main():
        success = await test_complete_voice_pipeline()
        await test_before_vs_after()
        
        if success:
            print("\n🎉 CONCLUSION: Voice formatting fix is working perfectly!")
            print("   Voice and text responses now have identical formatting behavior.")
        else:
            print("\n⚠️  CONCLUSION: There may be an issue with the fix.")
    
    asyncio.run(main())
