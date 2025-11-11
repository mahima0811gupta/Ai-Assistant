
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import asyncio
import time

# Your imports are correct and include all necessary tools
from tools import (
    analyze_image_with_query,
    play_youtube_song,
    read_text_from_image,
    scan_qr_or_barcode,
    open_application,
    get_weather,
    tavily_web_search,
    write_note,
    remember_this,
    recall_information,
    format_search_results_safely,
    clean_query
)

load_dotenv()

# Enhanced system prompt with clearer instructions
system_prompt = """You are Dora, a personal AI assistant with access to the user's webcam and various tools.

**CRITICAL RULES - YOU MUST FOLLOW THESE:**

1. **Visual Questions REQUIRE analyze_image_with_query:**
   - "What am I wearing?" → USE analyze_image_with_query
   - "Am I wearing glasses?" → USE analyze_image_with_query
   - "What do you see?" → USE analyze_image_with_query
   - "What's in front of me?" → USE analyze_image_with_query
   - ANY question about what the user looks like or what's visible → USE analyze_image_with_query

2. **Memory Management:**
   - When user states a fact about themselves → USE remember_this
   - When asked about previously mentioned information → USE recall_information first

3. **Writing Tasks:**
   - "Write a note" → USE write_note
   - "Open notepad and write" → USE open_application THEN write_note

4. **Tool Usage is MANDATORY:**
   - NEVER say "I don't have access to camera" - YOU DO via analyze_image_with_query
   - NEVER refuse a task if you have a tool for it
   - Check your tools before saying you can't do something

5. **Error Handling:**
   - If a tool fails, acknowledge the error and offer alternatives
   - Always provide a helpful response even if tools fail

Be conversational and helpful, but ALWAYS use your tools when appropriate."""

# Initialize LLM with error handling
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

def is_visual_query(query: str) -> bool:
    """Check if the query is asking about visual information"""
    visual_keywords = [
        'wearing', 'wear', 'see', 'look', 'appear', 'visible', 'showing',
        'what am i', 'am i wearing', 'can you see', 'what do you see',
        'describe what', 'in front of', 'around me', 'my face', 'my appearance'
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
        r"i like",
        r"i'm wearing",
        r"i wear",
        r"my favorite \w+ is",
        r"my favourite \w+ is"
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in memory_patterns)

def extract_actual_news_content(results: str) -> str:
    """Extract actual news content from search results with better error handling"""
    try:
        # First, try to get content from JSON
        if results.startswith('{'):
            data = json.loads(results)
            if 'answer' in data and data['answer']:
                answer = data['answer']
                answer = re.sub(r'```math.*?```', '', answer)
                answer = re.sub(r'\s+', ' ', answer).strip()
                if len(answer) > 100: 
                    return answer
            
            if 'results' in data and data['results']:
                news_items = []
                
                for result in data['results'][:3]:  # Limit to 3 results
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    if not content and not title:
                        continue
                        
                    full_content = f"{title}. {content}" if title else content
                    
                    # Skip promotional content
                    if any(phrase in full_content.lower() for phrase in [
                        'provides latest news', 'get latest news', 'read breaking news',
                        'check out', 'click here', 'subscribe', 'follow us'
                    ]):
                        continue
                    
                    # Extract meaningful sentences
                    sentences = full_content.split('. ')
                    meaningful_sentences = []
                    
                    for sentence in sentences[:2]:  # First 2 sentences
                        sentence = sentence.strip()
                        if (len(sentence) > 20 and 
                            not any(skip in sentence.lower() for skip in ['click', 'subscribe', 'website'])):
                            meaningful_sentences.append(sentence)
                    
                    if meaningful_sentences:
                        news_items.append('. '.join(meaningful_sentences))
                
                if news_items:
                    return ' '.join(news_items[:2])  # Return top 2 items
        
        # Fallback for string results
        if isinstance(results, str) and len(results) > 50:
            # Clean up the string
            cleaned = re.sub(r'```.*?```', '', results)
            cleaned = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if len(cleaned) > 100:
                return cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
        
        return "I found some search results but couldn't extract clear news stories. Try asking about specific topics."
        
    except Exception as e:
        print(f"Error extracting news: {e}")
        return "I had trouble processing the news results. Please try again."

def detect_query_intent(user_query: str) -> tuple:
    """Detect query intent and return (intent, should_use_direct_approach)"""
    query_lower = user_query.lower()
    
    # Check for visual queries first - these should go to the agent
    if is_visual_query(query_lower):
        return ('visual', False)
    
    # Check for memory statements - these should go to the agent
    if is_memory_statement(query_lower):
        return ('memory', False)
    
    # Check for compound commands
    if ("open" in query_lower or "launch" in query_lower) and ("write" in query_lower or "type" in query_lower):
        return ('compound_command', False)
    
    # Music keywords detection
    music_keywords = ['play', 'song', 'music']
    if any(keyword in query_lower for keyword in music_keywords):
        if not any(word in query_lower for word in ['game', 'video', 'movie']):
            return ('music', True)
    
    # Application keywords detection
    app_keywords = ['open', 'launch', 'start', 'begin', 'run']
    apps = ['notepad', 'chrome', 'calculator', 'explorer', 'browser']
    if any(keyword in query_lower for keyword in app_keywords) and any(app in query_lower for app in apps):
        return ('application', True)

    # Weather keywords
    weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy']
    if any(keyword in query_lower for keyword in weather_keywords):
        return ('weather', True)
    
    # News keywords
    news_keywords = ['news', 'latest', 'headlines', 'breaking', 'current events']
    if any(keyword in query_lower for keyword in news_keywords):
        return ('news', True)
    
    # Cricket/sports keywords
    cricket_keywords = ['cricket', 'score', 'match', 'game']
    if any(keyword in query_lower for keyword in cricket_keywords):
        return ('cricket', True)
    
    return ('general', False)

def extract_song_name(user_query: str) -> str:
    """Better extraction of song name from query"""
    query_lower = user_query.lower()
    
    # Pattern matching for different formats
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
    
    # Fallback: remove common prefixes
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
            print(f"Connection reset error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(1)  # Wait 1 second before retry
                continue
            return f"Connection error occurred. Please try again."
        except Exception as e:
            print(f"Tool execution error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(0.5)
                continue
            return f"Tool error: {str(e)}"
    
    return "Tool execution failed after retries."

def direct_tool_handler(intent: str, user_query: str) -> str:
    """Handle specific queries directly with tools - now with better error handling"""
    query_lower = user_query.lower()
    
    if intent == 'music':
        try:
            song_name = extract_song_name(user_query)
            if not song_name:
                return "I couldn't understand which song you want me to play. Please specify the song name."
            print(f"🎵 Directly playing song: '{song_name}'")
            return safe_tool_execution(play_youtube_song._run, song_name)
        except Exception as e:
            return f"Error playing song: {e}"

    elif intent == 'application':
        try:
            app_mapping = {
                'chrome': ['chrome', 'browser', 'google chrome'],
                'notepad': ['notepad', 'text editor'],
                'calculator': ['calculator', 'calc'],
                'explorer': ['explorer', 'file explorer', 'files']
            }
            
            app_name = ""
            for app, keywords in app_mapping.items():
                if any(keyword in query_lower for keyword in keywords):
                    app_name = app
                    break
                    
            if app_name:
                print(f"🚀 Directly opening application: '{app_name}'")
                return safe_tool_execution(open_application._run, app_name)
            return "Could not determine which application to open."
        except Exception as e:
            return f"Error opening application: {e}"

    elif intent == 'news':
        try:
            print(f"🔍 Searching for news: {user_query}")
            enhanced_query = "latest India news India current events today breaking news stories headlines"
            raw_results = safe_tool_execution(tavily_web_search._run, enhanced_query)
            return extract_actual_news_content(raw_results)
        except Exception as e:
            return f"Sorry, there was an error searching for news: {e}"
    
    elif intent == 'cricket':
        try:
            print(f"🏏 Searching for cricket: {user_query}")
            raw_results = safe_tool_execution(tavily_web_search._run, user_query)
            return extract_actual_news_content(raw_results)
        except Exception as e:
            return f"Sorry, there was an error searching for cricket information: {e}"
    
    elif intent == 'weather':
        try:
            print(f"🌤️ Getting weather for: {user_query}")
            result = safe_tool_execution(get_weather._run, user_query)
            # Clean up weather response
            result = re.sub(r'^#+\s*', '', str(result))
            result = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', result)
            return result.strip()
        except Exception as e:
            return f"Sorry, there was an error getting the weather: {e}"
            
    return "Direct handler could not process the request."

def ask_agent(user_query: str, chat_history: list) -> str:
    """
    Processes a user query with improved error handling and connection stability.
    """
    if not llm:
        return "Sorry, the AI model is not available. Please check your configuration."
    
    print(f"🧠 Agent received query: {user_query}")
    
    # Clean the query first to remove audio artifacts
    cleaned_query = clean_query(user_query)
    print(f"🧹 Cleaned query: '{cleaned_query}'")
    
    # If the query becomes empty or too short after cleaning, return a friendly message
    if not cleaned_query.strip() or len(cleaned_query.strip()) < 3:
        return "I'm listening, but I only heard some background sounds. Could you please speak more clearly?"
    
    intent, use_direct = detect_query_intent(cleaned_query)
    
    if use_direct:
        print(f"🎯 Using direct approach for '{intent}' query")
        return direct_tool_handler(intent, cleaned_query)
    
    print("🤖 Using LangGraph agent for general query")
    
    try:
        # Create the agent with error handling
        agent = create_react_agent(
            llm,
            [
                analyze_image_with_query,
                remember_this,
                recall_information,
                tavily_web_search,
                get_weather,
                play_youtube_song,
                read_text_from_image,
                scan_qr_or_barcode,
                open_application,
                write_note
            ]
        )
        
        # Add system message to the conversation
        conversation = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": cleaned_query}]
        input_messages = {"messages": conversation}
        
        # Execute with timeout and error handling
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = agent.invoke(input_messages)
                final_content = response['messages'][-1].content
                print(f"✅ Agent returned response: {final_content}")
                
                # Handle memory statements
                if intent == 'memory':
                    try:
                        # Also save to memory if it's a personal statement
                        memory_result = safe_tool_execution(remember_this._run, cleaned_query)
                        print(f"💾 Saved to memory: {memory_result}")
                    except Exception as e:
                        print(f"Memory save failed: {e}")
                
                return final_content
            
            except ConnectionResetError as e:
                print(f"Connection reset on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(2)  # Wait longer for connection issues
                    continue
                return "I'm having connection issues. Please try asking again."
            
            except Exception as e:
                print(f"Agent error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return f"I encountered an error processing your request. Please try again."
    
    except Exception as e:
        print(f"❌ Critical error in agent processing: {e}")
        return "Sorry, I encountered a critical error. Please restart the application if this persists."