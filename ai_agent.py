
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import time
from datetime import datetime

from tools import (
    play_youtube_song,
    open_application,
    get_weather,
    tavily_web_search,
    write_note,
    remember_this,
    recall_information,
    get_current_time,
    clean_query
)

load_dotenv()

# Enhanced system prompt
system_prompt = """You are Dora, an intelligent personal AI voice assistant created to help users with various tasks.

**YOUR CAPABILITIES:**
- Play music on YouTube
- Get weather information for any city
- Search the web for latest news and information
- Open system applications (Chrome, Notepad, Calculator, etc.)
- Write and save notes
- Remember and recall user information
- Tell current time and date

**CRITICAL RULES:**

1. **NO VISUAL CAPABILITIES:**
   - You do NOT have camera or webcam access
   - You CANNOT see what the user is wearing, their appearance, or their surroundings
   - If asked about visual information, politely explain you're a voice-only assistant

2. **Memory Management:**
   - When user states facts about themselves (name, preferences, etc.) → USE remember_this tool
   - When asked about previously mentioned information → USE recall_information tool first
   - Examples: "My name is...", "I like...", "My favorite... is..."

3. **Tool Usage Priority:**
   - ALWAYS check your available tools before saying you can't do something
   - Use appropriate tools for: music, weather, web search, apps, notes, memory, time
   - Provide tool results directly without unnecessary explanations

4. **Compound Commands:**
   - For "open notepad and write..." commands, use BOTH open_application AND write_note tools
   - First open the application, then write the content

5. **Response Style:**
   - Be conversational, friendly, and concise
   - Avoid unnecessary apologies or over-explaining
   - Get straight to the point
   - If a tool fails, acknowledge it briefly and offer an alternative

6. **Error Handling:**
   - If a tool returns an error, inform the user clearly
   - Suggest alternatives when possible
   - Never blame the user for errors

**EXAMPLES:**

User: "Play Shape of You"
You: [Use play_youtube_song] → "Now playing Shape of You on YouTube."

User: "What's the weather in Delhi?"
You: [Use get_weather] → "Weather in Delhi: 28°C, clear sky with 45% humidity."

User: "My name is Raj"
You: [Use remember_this] → "Got it! I'll remember that your name is Raj."

User: "What's my name?"
You: [Use recall_information] → "Your name is Raj."

User: "Open notepad and write hello everyone"
You: [Use open_application, then write_note] → "I've opened notepad and written 'hello everyone'."

Be helpful, efficient, and always use your tools when appropriate!"""

# Initialize LLM with Gemini 2.5 Flash
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )
    print("✅ LLM initialized successfully with Gemini 2.5 Flash")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    llm = None


def is_visual_query(query: str) -> bool:
    """Check if the query is asking about visual information"""
    visual_keywords = [
        'wearing', 'wear', 'see', 'look', 'appear', 'visible', 'showing',
        'what am i', 'am i wearing', 'can you see', 'what do you see',
        'describe what', 'in front of', 'around me', 'my face', 'my appearance',
        'camera', 'webcam', 'picture', 'photo', 'image', 'looking at'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in visual_keywords)


def is_memory_statement(query: str) -> bool:
    """Check if the user is stating a fact about themselves"""
    memory_patterns = [
        r"i am \w+",
        r"i'm \w+",
        r"i have \w+",
        r"i've \w+",
        r"my name is",
        r"call me \w+",
        r"i like",
        r"i love",
        r"i prefer",
        r"my favorite \w+ is",
        r"my favourite \w+ is",
        r"i work (at|as|in)",
        r"i live in"
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in memory_patterns)


def extract_actual_news_content(results: str) -> str:
    """Extract meaningful news content from search results"""
    try:
        if results.startswith('{'):
            data = json.loads(results)
            
            # Try to get direct answer first
            if 'answer' in data and data['answer']:
                answer = data['answer']
                # Clean up markdown and extra formatting
                answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL)
                answer = re.sub(r'\s+', ' ', answer).strip()
                if len(answer) > 100:
                    return answer
            
            # Extract from results
            if 'results' in data and data['results']:
                news_items = []
                
                for result in data['results'][:3]:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    if not content and not title:
                        continue
                    
                    # Skip promotional content
                    skip_phrases = [
                        'provides latest news', 'get latest news', 'read breaking news',
                        'click here', 'subscribe', 'follow us', 'visit our website'
                    ]
                    
                    full_content = f"{title}. {content}" if title else content
                    
                    if any(phrase in full_content.lower() for phrase in skip_phrases):
                        continue
                    
                    # Extract meaningful sentences
                    sentences = full_content.split('. ')
                    meaningful_sentences = [
                        s.strip() for s in sentences[:3]
                        if len(s.strip()) > 30 and not any(skip in s.lower() for skip in skip_phrases)
                    ]
                    
                    if meaningful_sentences:
                        news_items.append('. '.join(meaningful_sentences))
                
                if news_items:
                    combined = ' '.join(news_items[:2])
                    return combined[:600] + "..." if len(combined) > 600 else combined
        
        # Fallback: clean up raw string results
        if isinstance(results, str) and len(results) > 50:
            cleaned = re.sub(r'```.*?```', '', results, flags=re.DOTALL)
            cleaned = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if len(cleaned) > 100:
                return cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
        
        return "I found some search results but couldn't extract clear information. Try asking about specific topics."
        
    except Exception as e:
        print(f"❌ Error extracting news: {e}")
        return "I had trouble processing the search results. Please try again."


def detect_query_intent(user_query: str) -> tuple:
    """Detect query intent and return (intent, should_use_direct_approach)"""
    query_lower = user_query.lower()
    
    # Check for visual queries first
    if is_visual_query(query_lower):
        return ('visual_rejected', True)
    
    # Check for memory statements
    if is_memory_statement(query_lower):
        return ('memory', False)
    
    # Check for time queries
    time_keywords = ['time', 'date', 'what day', 'current time', "what's the time"]
    if any(keyword in query_lower for keyword in time_keywords):
        return ('time', True)
    
    # Check for compound commands (open + write) - Let agent handle it
    if ('open' in query_lower or 'launch' in query_lower) and ('write' in query_lower or 'type' in query_lower):
        return ('compound_command', False)  # False = Let agent handle it intelligently
    
    # Check for music (but not "open youtube")
    music_keywords = ['play', 'song', 'music']
    if any(keyword in query_lower for keyword in music_keywords):
        if not any(word in query_lower for word in ['game', 'video', 'movie']):
            # Don't treat "open youtube" as music
            if 'open' in query_lower and 'youtube' in query_lower:
                return ('application', True)
            return ('music', True)
    
    # Check for applications
    app_keywords = ['open', 'launch', 'start', 'run']
    apps = ['notepad', 'chrome', 'calculator', 'explorer', 'browser', 'paint', 'youtube']
    if any(keyword in query_lower for keyword in app_keywords) and any(app in query_lower for app in apps):
        return ('application', True)

    # Check for weather
    weather_keywords = ['weather', 'temperature', 'temp', 'forecast', 'rain', 'sunny', 'cloudy']
    if any(keyword in query_lower for keyword in weather_keywords):
        return ('weather', True)
    
    # Check for news
    news_keywords = ['news', 'latest', 'headlines', 'breaking', 'current events']
    if any(keyword in query_lower for keyword in news_keywords):
        return ('news', True)
    
    return ('general', False)


def extract_song_name(user_query: str) -> str:
    """Extract song name from query"""
    query_lower = user_query.lower()
    
    # Try pattern matching
    patterns = [
        r"play (?:the )?(?:song )?['\"]?(.+?)['\"]? by (.+?)(?:\s+on youtube)?$",
        r"play (.+) by (.+)",
        r"play (?:the )?(?:song )?['\"]?(.+?)['\"]?(?:\s+on youtube)?$",
        r"play (.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                return f"{match.group(1)} {match.group(2)}".strip()
            else:
                return match.group(1).strip()
    
    # Remove common prefixes
    prefixes = ['play song', 'play the song', 'play music', 'play']
    result = query_lower
    for prefix in prefixes:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
            break
    
    return result.strip()


def safe_tool_execution(tool_func, *args, **kwargs):
    """Safely execute a tool function with error handling and retries"""
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            result = tool_func(*args, **kwargs)
            return result
        except ConnectionResetError as e:
            print(f"⚠️ Connection reset error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            return "Connection error occurred. Please try again."
        except Exception as e:
            print(f"⚠️ Tool execution error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return f"Tool error: {str(e)}"
    
    return "Tool execution failed after retries."


def direct_tool_handler(intent: str, user_query: str) -> str:
    """Handle specific queries directly with tools"""
    query_lower = user_query.lower()
    
    if intent == 'visual_rejected':
        return ("I'm a voice-only assistant and don't have access to a camera or webcam. "
                "I can help you with music, weather, web searches, opening apps, writing notes, "
                "time, and remembering information. What would you like me to do?")
    
    if intent == 'time':
        try:
            print(f"⏰ Getting current time")
            return safe_tool_execution(get_current_time._run, "12h")
        except Exception as e:
            return f"Error getting time: {e}"
    
    if intent == 'music':
        try:
            song_name = extract_song_name(user_query)
            if not song_name or len(song_name) < 2:
                return "I couldn't understand which song you want me to play. Please specify the song name clearly."
            print(f"🎵 Directly playing song: '{song_name}'")
            return safe_tool_execution(play_youtube_song._run, song_name)
        except Exception as e:
            return f"Error playing song: {e}"

    elif intent == 'application':
        try:
            # Handle "open youtube" case
            if 'youtube' in query_lower:
                print(f"🌐 Opening YouTube in Chrome")
                return safe_tool_execution(open_application._run, 'chrome')
            
            app_mapping = {
                'chrome': ['chrome', 'browser', 'google chrome'],
                'notepad': ['notepad', 'text editor'],
                'calculator': ['calculator', 'calc'],
                'explorer': ['explorer', 'file explorer', 'files'],
                'paint': ['paint', 'mspaint']
            }
            
            app_name = ""
            for app, keywords in app_mapping.items():
                if any(keyword in query_lower for keyword in keywords):
                    app_name = app
                    break
                    
            if app_name:
                print(f"🚀 Directly opening application: '{app_name}'")
                return safe_tool_execution(open_application._run, app_name)
            return "Could not determine which application to open. Try: chrome, notepad, calculator, explorer, or paint."
        except Exception as e:
            return f"Error opening application: {e}"

    elif intent == 'news':
        try:
            print(f"📰 Searching for news")
            if 'india' in query_lower or 'indian' in query_lower:
                enhanced_query = "latest India news today breaking headlines current events"
            else:
                enhanced_query = f"{user_query} latest news today"
            
            raw_results = safe_tool_execution(tavily_web_search._run, enhanced_query)
            return extract_actual_news_content(raw_results)
        except Exception as e:
            return f"Sorry, there was an error searching for news: {e}"
    
    elif intent == 'weather':
        try:
            # Pass full query to weather tool - let it extract the city
            print(f"🌤️ Getting weather for: {user_query}")
            result = safe_tool_execution(get_weather._run, user_query)
            # Clean up formatting
            result = re.sub(r'^#+\s*', '', str(result))
            result = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', result)
            return result.strip()
        except Exception as e:
            return f"Sorry, there was an error getting the weather: {e}"
            
    return "I couldn't process that request. Please try again."


def ask_agent(user_query: str, chat_history: list = None) -> str:
    """
    Process user query with enhanced error handling and direct tool execution
    
    Args:
        user_query: The user's question or command
        chat_history: Previous conversation history (optional)
    
    Returns:
        Response string
    """
    if chat_history is None:
        chat_history = []
    
    if not llm:
        return "Sorry, the AI model is not available. Please check your API key configuration."
    
    print(f"\n{'='*60}")
    print(f"🧠 Agent received query: {user_query}")
    print(f"{'='*60}")
    
    # Clean the query
    cleaned_query = clean_query(user_query)
    print(f"🧹 Cleaned query: '{cleaned_query}'")
    
    # Validate query
    if not cleaned_query.strip() or len(cleaned_query.strip()) < 2:
        return "I'm listening, but I didn't catch that clearly. Could you please repeat?"
    
    # Detect intent
    intent, use_direct = detect_query_intent(cleaned_query)
    print(f"🎯 Detected intent: '{intent}' | Direct approach: {use_direct}")
    
    # Use direct handler for specific intents
    if use_direct:
        print(f"⚡ Using direct tool handler")
        return direct_tool_handler(intent, cleaned_query)
    
    # Use LangGraph agent for complex queries
    print("🤖 Using LangGraph agent")
    
    try:
        # Create agent with all tools
        agent = create_react_agent(
            llm,
            [
                remember_this,
                recall_information,
                tavily_web_search,
                get_weather,
                play_youtube_song,
                open_application,
                write_note,
                get_current_time
            ]
        )
        
        # Prepare conversation
        conversation = [
            {"role": "system", "content": system_prompt}
        ] + chat_history + [
            {"role": "user", "content": cleaned_query}
        ]
        
        input_messages = {"messages": conversation}
        
        # Execute with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = agent.invoke(input_messages)
                final_content = response['messages'][-1].content
                print(f"✅ Agent response: {final_content[:100]}...")
                
                # Auto-save memory for memory statements
                if intent == 'memory':
                    try:
                        memory_result = safe_tool_execution(remember_this._run, cleaned_query)
                        print(f"💾 Auto-saved to memory: {memory_result}")
                    except Exception as e:
                        print(f"⚠️ Memory save failed: {e}")
                
                return final_content
            
            except ConnectionResetError as e:
                print(f"⚠️ Connection reset on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return "I'm having connection issues. Please try asking again in a moment."
            
            except Exception as e:
                print(f"⚠️ Agent error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return "I encountered an error processing your request. Please try rephrasing your question."
    
    except Exception as e:
        print(f"❌ Critical error in agent processing: {e}")
        return "Sorry, I encountered a critical error. Please try again or restart the application."