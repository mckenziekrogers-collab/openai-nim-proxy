// server.js - OpenAI to NVIDIA NIM API Proxy (ULTRA LONG CHAT + FORMAT ENFORCEMENT)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// 🔥 MAXIMUM payload limits to handle massive conversations
app.use(cors());
app.use(express.json({ limit: '500mb' }));
app.use(express.urlencoded({ limit: '500mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🎭 FORMAT ENFORCEMENT - Forces consistent RP format
const ENFORCE_FORMAT = true; // Set to false to disable format matching
const FORMAT_STRICTNESS = 'high'; // 'low', 'medium', 'high'

// 🔥 EXTREME CONTEXT MANAGEMENT - For 500-2000+ message conversations
const MAX_CONTEXT_MESSAGES = 30; // Maximum messages sent to API
const PRESERVE_RECENT_MESSAGES = 20; // Keep last N messages untouched
const SMART_COMPRESSION = true; // Use intelligent compression
const EMERGENCY_TRUNCATE = 50000; // Emergency token limit before hard truncation

// Model mapping - Best DeepSeek models
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
  'gpt-4': 'deepseek-ai/deepseek-v3.1',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'claude-3-sonnet': 'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'gemini-pro': 'deepseek-ai/deepseek-r1-distill-qwen-7b'
};

// Helper: Estimate tokens (4 chars ≈ 1 token)
function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

// Helper: Get total tokens from messages
function getTotalTokens(messages) {
  return messages.reduce((sum, msg) => {
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    return sum + estimateTokens(content);
  }, 0);
}

// Helper: Detect user's formatting style
function detectFormatStyle(messages) {
  const userMessages = messages.filter(m => m.role === 'user').slice(-5);
  
  let hasAsterisks = false;
  let hasQuotes = false;
  let exampleFormat = '';
  
  for (const msg of userMessages) {
    const content = msg.content || '';
    
    // Check for *action* format
    if (/\*[^*]+\*/.test(content)) hasAsterisks = true;
    
    // Check for "dialogue" format
    if (/"[^"]+"/g.test(content)) hasQuotes = true;
    
    // Extract example
    if (hasAsterisks && hasQuotes && !exampleFormat) {
      const lines = content.split('\n');
      for (const line of lines) {
        if (line.includes('*') && line.includes('"')) {
          exampleFormat = line.trim();
          break;
        }
      }
    }
  }
  
  return {
    usesRPFormat: hasAsterisks && hasQuotes,
    exampleFormat
  };
}

// Helper: Create format enforcement instruction
function createFormatInstruction(formatStyle, strictness) {
  if (!formatStyle.usesRPFormat) return '';
  
  const strictnessMap = {
    low: 'Try to match',
    medium: 'Please match',
    high: 'You MUST strictly match'
  };
  
  const prefix = strictnessMap[strictness] || strictnessMap.medium;
  
  let instruction = `\n\n[CRITICAL FORMATTING REQUIREMENT]:\n${prefix} this exact writing style:\n`;
  instruction += `• Actions/descriptions: *Use asterisks* like this: *character leans forward*\n`;
  instruction += `• Dialogue: "Use quotation marks" like this: "Hello there"\n`;
  instruction += `• Mix them naturally: *Sarah smiles.* "I've been waiting for you." *She gestures to a chair.*\n`;
  
  if (formatStyle.exampleFormat) {
    instruction += `\nUser's style example:\n${formatStyle.exampleFormat}\n`;
  }
  
  if (strictness === 'high') {
    instruction += `\n❌ DO NOT use these formats:\n`;
    instruction += `- Plain prose without asterisks\n`;
    instruction += `- (Parentheses for actions)\n`;
    instruction += `- Mixed formats or inconsistent styling\n`;
    instruction += `\n✅ ALWAYS use *asterisks* for actions and "quotes" for dialogue throughout your entire response.\n`;
  }
  
  return instruction;
}

// Helper: Aggressive smart truncation for extreme message counts
function aggressiveTruncate(messages, formatInstruction = '') {
  console.log(`⚡ AGGRESSIVE TRUNCATION: ${messages.length} messages`);
  
  // Separate system message
  const systemMsg = messages.find(m => m.role === 'system');
  const otherMessages = messages.filter(m => m.role !== 'system');
  
  // Always keep recent messages
  const recentMessages = otherMessages.slice(-PRESERVE_RECENT_MESSAGES);
  const olderMessages = otherMessages.slice(0, -PRESERVE_RECENT_MESSAGES);
  
  console.log(`   Preserving ${recentMessages.length} recent messages`);
  console.log(`   Compressing ${olderMessages.length} older messages`);
  
  // Create ultra-compact summary of old messages
  const summaryChunks = [];
  const chunkSize = 50; // Compress in 50-message chunks
  
  for (let i = 0; i < olderMessages.length; i += chunkSize) {
    const chunk = olderMessages.slice(i, i + chunkSize);
    
    // Extract only critical info from chunk
    const summary = chunk.map(m => {
      const preview = m.content.substring(0, 100);
      return `${m.role}:${preview}`;
    }).join('|');
    
    summaryChunks.push(summary);
  }
  
  const fullSummary = `[${olderMessages.length} earlier messages compressed]: ${summaryChunks.join(' >> ')}`;
  
  const result = [];
  
  // Add system message with format instruction
  if (systemMsg) {
    result.push({
      role: 'system',
      content: systemMsg.content + (formatInstruction || '')
    });
  } else if (formatInstruction) {
    result.push({
      role: 'system',
      content: `You are a creative writing assistant.${formatInstruction}`
    });
  }
  
  // Add compact summary
  if (olderMessages.length > 0) {
    result.push({
      role: 'system',
      content: fullSummary
    });
  }
  
  // Add recent messages
  result.push(...recentMessages);
  
  const originalTokens = getTotalTokens(messages);
  const finalTokens = getTotalTokens(result);
  
  console.log(`   ${messages.length} → ${result.length} messages`);
  console.log(`   ~${originalTokens} → ~${finalTokens} tokens (${((1 - finalTokens/originalTokens) * 100).toFixed(1)}% reduction)`);
  
  return result;
}

// Main compression function
function compressMessages(messages, formatInstruction = '') {
  const totalMessages = messages.length;
  const totalTokens = getTotalTokens(messages);
  
  console.log(`\n📊 CONTEXT ANALYSIS:`);
  console.log(`   Messages: ${totalMessages}`);
  console.log(`   Tokens: ~${totalTokens}`);
  
  // No compression needed
  if (totalMessages <= MAX_CONTEXT_MESSAGES && totalTokens < EMERGENCY_TRUNCATE) {
    console.log(`✅ Under limits, no compression needed\n`);
    
    // Still add format instruction
    if (formatInstruction) {
      const systemMsg = messages.find(m => m.role === 'system');
      const result = [...messages];
      
      if (systemMsg) {
        const idx = result.findIndex(m => m.role === 'system');
        result[idx] = {
          ...systemMsg,
          content: systemMsg.content + formatInstruction
        };
      } else {
        result.unshift({
          role: 'system',
          content: `You are a creative writing assistant.${formatInstruction}`
        });
      }
      
      return result;
    }
    
    return messages;
  }
  
  // Apply smart compression
  console.log(`🔥 Applying smart compression...`);
  return aggressiveTruncate(messages, formatInstruction);
}

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Ultra Long Chat Proxy with Format Enforcement',
    format_enforcement: ENFORCE_FORMAT,
    format_strictness: FORMAT_STRICTNESS,
    max_context_messages: MAX_CONTEXT_MESSAGES,
    preserve_recent: PRESERVE_RECENT_MESSAGES,
    payload_limit: '500mb',
    optimized_for: 'Fanfic RP with 500-2000+ messages'
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Main chat endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    console.log(`\n📨 NEW REQUEST: ${model} | ${messages.length} messages`);
    
    // Detect and enforce format
    let formatInstruction = '';
    if (ENFORCE_FORMAT) {
      const formatStyle = detectFormatStyle(messages);
      if (formatStyle.usesRPFormat) {
        formatInstruction = createFormatInstruction(formatStyle, FORMAT_STRICTNESS);
        console.log(`🎭 Format enforcement: ACTIVE (${FORMAT_STRICTNESS})`);
      }
    }
    
    // Compress messages
    const compressedMessages = compressMessages(messages, formatInstruction);
    
    // Select model
    let nimModel = MODEL_MAPPING[model] || 'deepseek-ai/deepseek-v3.1';
    console.log(`🤖 Using: ${nimModel}\n`);
    
    // Build request
    const nimRequest = {
      model: nimModel,
      messages: compressedMessages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    // Call API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000, // 5 min timeout
      maxContentLength: 500 * 1024 * 1024, // 500MB
      maxBodyLength: 500 * 1024 * 1024
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.pipe(res);
      
      response.data.on('end', () => {
        console.log(`✅ Stream completed\n`);
      });
      
      response.data.on('error', (err) => {
        console.error(`❌ Stream error:`, err.message);
        res.end();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices,
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      console.log(`✅ Request completed\n`);
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error(`\n❌ ERROR:`, error.message);
    
    // Handle payload too large error
    if (error.code === 'ERR_HTTP_INVALID_STATUS_CODE' || 
        error.message.includes('payload') || 
        error.message.includes('too large')) {
      console.error(`💥 PAYLOAD TOO LARGE - Try reducing MAX_CONTEXT_MESSAGES or PRESERVE_RECENT_MESSAGES`);
    }
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// 404 handler
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n╔═══════════════════════════════════════════════════════╗`);
  console.log(`║  🚀 ULTRA LONG CHAT PROXY + FORMAT ENFORCEMENT      ║`);
  console.log(`╚═══════════════════════════════════════════════════════╝`);
  console.log(`\n📡 Port: ${PORT}`);
  console.log(`🏥 Health: http://localhost:${PORT}/health`);
  console.log(`\n⚙️  CONFIG:`);
  console.log(`   • Payload limit: 500MB`);
  console.log(`   • Max messages: ${MAX_CONTEXT_MESSAGES}`);
  console.log(`   • Preserve recent: ${PRESERVE_RECENT_MESSAGES}`);
  console.log(`   • Format enforcement: ${ENFORCE_FORMAT ? '✅' : '❌'} (${FORMAT_STRICTNESS})`);
  console.log(`\n💪 OPTIMIZED FOR:`);
  console.log(`   • 500-2000+ message fanfic RP chats`);
  console.log(`   • Consistent *action* "dialogue" format`);
  console.log(`   • DeepSeek models (free unlimited)`);
  console.log(`\n🎭 Format enforcement automatically matches your style!`);
  console.log(`🔥 Ready for extreme conversation lengths!\n`);
});
