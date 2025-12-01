// server.js - OpenAI to NVIDIA NIM API Proxy (EXTREME LONG CHAT OPTIMIZED)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware with MASSIVE payload limit
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 EXTREME CONTEXT MANAGEMENT - Optimized for 500-1000+ message conversations
const MAX_CONTEXT_MESSAGES = 25; // Final context window sent to API
const SUMMARIZATION_TRIGGER = 20; // Start summarizing when old messages exceed this
const PRESERVE_SYSTEM_PROMPT = true; // Always keep system message
const PRESERVE_RECENT_MESSAGES = 15; // Keep last N messages completely intact (NO summarization)
const AGGRESSIVE_COMPRESSION = true; // Multi-stage compression for 100+ messages
const ROLLING_SUMMARY_SIZE = 30; // Chunk size for parallel summarization

// Model mapping - DeepSeek models that actually work!
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
  'gpt-4': 'deepseek-ai/deepseek-v3.1',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'deepseek-v3': 'deepseek-ai/deepseek-v3.1',
  'deepseek-r1': 'deepseek-ai/deepseek-r1',
  'deepseek-coder': 'deepseek-ai/deepseek-coder-6.7b-instruct',
  'claude-3-opus': 'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'claude-3-sonnet': 'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'gemini-pro': 'deepseek-ai/deepseek-r1-distill-qwen-7b'
};

// Helper: Count tokens (rough estimate: ~4 chars = 1 token)
function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

// Helper: Estimate total tokens in messages array
function estimateTotalTokens(messages) {
  return messages.reduce((total, msg) => {
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    return total + estimateTokens(content);
  }, 0);
}

// Helper: Create summary of a message chunk using AI
async function createChunkSummary(messages, chunkIndex, totalChunks) {
  try {
    const conversationText = messages.map(m => `${m.role}: ${m.content}`).join('\n\n');
    
    const summaryPrompt = `You are summarizing part ${chunkIndex + 1} of ${totalChunks} from a long fanfic planning conversation. Extract ONLY the most critical information: character names, relationships, plot points, decisions made, worldbuilding details, story arcs, and any creative choices. Be extremely concise but preserve all important details.

Conversation segment:
${conversationText}

Provide a focused summary (max 300 words):`;
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
      model: 'deepseek-ai/deepseek-r1-distill-qwen-7b', // Fast DeepSeek model for summaries
      messages: [{ role: 'user', content: summaryPrompt }],
      max_tokens: 400,
      temperature: 0.2
    }, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 15000 // 15 second timeout per summary
    });
    
    return response.data.choices[0].message.content;
  } catch (error) {
    console.error(`⚠️ Summary creation failed for chunk ${chunkIndex}:`, error.message);
    // Fallback: grab key parts from each message
    return messages.map(m => {
      const preview = m.content.substring(0, 150);
      return `${m.role}: ${preview}...`;
    }).join(' | ');
  }
}

// Helper: AGGRESSIVE multi-stage compression for 100+ messages
async function aggressiveCompression(messages) {
  console.log(`\n🔥 AGGRESSIVE COMPRESSION activated for ${messages.length} messages`);
  
  const systemMessage = PRESERVE_SYSTEM_PROMPT ? messages.find(m => m.role === 'system') : null;
  const nonSystemMessages = messages.filter(m => m.role !== 'system');
  
  // ALWAYS preserve the most recent messages completely intact
  const recentMessages = nonSystemMessages.slice(-PRESERVE_RECENT_MESSAGES);
  const olderMessages = nonSystemMessages.slice(0, -PRESERVE_RECENT_MESSAGES);
  
  console.log(`📊 Breakdown: Recent (preserved) = ${recentMessages.length}, Older (to compress) = ${olderMessages.length}`);
  
  if (olderMessages.length === 0) {
    const result = systemMessage ? [systemMessage, ...recentMessages] : recentMessages;
    console.log(`✅ No old messages to compress, returning ${result.length} messages\n`);
    return result;
  }
  
  // Split older messages into chunks for parallel summarization
  const chunks = [];
  for (let i = 0; i < olderMessages.length; i += ROLLING_SUMMARY_SIZE) {
    chunks.push(olderMessages.slice(i, i + ROLLING_SUMMARY_SIZE));
  }
  
  console.log(`📦 Split into ${chunks.length} chunks of ~${ROLLING_SUMMARY_SIZE} messages each`);
  console.log(`⚡ Summarizing chunks in parallel...`);
  
  // Summarize all chunks in parallel for speed
  const summaryPromises = chunks.map((chunk, index) => 
    createChunkSummary(chunk, index, chunks.length)
  );
  
  const summaries = await Promise.all(summaryPromises);
  
  // Combine all chunk summaries into one condensed history
  const combinedSummary = summaries
    .map((summary, i) => `[Section ${i + 1}/${summaries.length}]:\n${summary}`)
    .join('\n\n');
  
  const summaryMessage = {
    role: 'system',
    content: `[Condensed conversation history - ${olderMessages.length} messages summarized]:\n\n${combinedSummary}`
  };
  
  // Build final compressed context
  const compressed = [];
  if (systemMessage) compressed.push(systemMessage);
  compressed.push(summaryMessage);
  compressed.push(...recentMessages);
  
  const originalTokens = estimateTotalTokens(messages);
  const compressedTokens = estimateTotalTokens(compressed);
  
  console.log(`✅ Compression complete!`);
  console.log(`   ${messages.length} messages → ${compressed.length} messages`);
  console.log(`   ~${originalTokens} tokens → ~${compressedTokens} tokens`);
  console.log(`   Compression ratio: ${((1 - compressedTokens / originalTokens) * 100).toFixed(1)}%\n`);
  
  return compressed;
}

// Helper: Standard compression for moderate message counts (20-100 messages)
async function standardCompression(messages) {
  console.log(`\n📝 Standard compression for ${messages.length} messages`);
  
  const systemMessage = PRESERVE_SYSTEM_PROMPT ? messages.find(m => m.role === 'system') : null;
  const nonSystemMessages = messages.filter(m => m.role !== 'system');
  
  const recentMessages = nonSystemMessages.slice(-PRESERVE_RECENT_MESSAGES);
  const olderMessages = nonSystemMessages.slice(0, -PRESERVE_RECENT_MESSAGES);
  
  if (olderMessages.length === 0 || olderMessages.length <= SUMMARIZATION_TRIGGER) {
    // Just trim, no summary needed
    const trimmed = [];
    if (systemMessage) trimmed.push(systemMessage);
    trimmed.push(...nonSystemMessages.slice(-MAX_CONTEXT_MESSAGES));
    console.log(`✂️ Simple trim: ${messages.length} → ${trimmed.length} messages\n`);
    return trimmed;
  }
  
  // Create single summary for all older messages
  console.log(`📄 Creating single summary for ${olderMessages.length} older messages...`);
  const summary = await createChunkSummary(olderMessages, 0, 1);
  
  const summaryMessage = {
    role: 'system',
    content: `[Previous conversation summary - ${olderMessages.length} messages]:\n\n${summary}`
  };
  
  const compressed = [];
  if (systemMessage) compressed.push(systemMessage);
  compressed.push(summaryMessage);
  compressed.push(...recentMessages);
  
  console.log(`✅ Standard compression: ${messages.length} → ${compressed.length} messages\n`);
  return compressed;
}

// Main compression router - chooses strategy based on message count
async function compressContext(messages) {
  const totalMessages = messages.length;
  const estimatedTokens = estimateTotalTokens(messages);
  
  console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  console.log(`🔍 CONTEXT ANALYSIS`);
  console.log(`   Messages: ${totalMessages}`);
  console.log(`   Estimated tokens: ~${estimatedTokens}`);
  console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
  
  // No compression needed
  if (totalMessages <= MAX_CONTEXT_MESSAGES) {
    console.log(`✅ Under limit (${MAX_CONTEXT_MESSAGES}), no compression needed\n`);
    return messages;
  }
  
  // Choose compression strategy
  if (AGGRESSIVE_COMPRESSION && totalMessages > 100) {
    console.log(`🚀 Strategy: AGGRESSIVE (${totalMessages} messages > 100)`);
    return await aggressiveCompression(messages);
  } else {
    console.log(`📝 Strategy: STANDARD (${totalMessages} messages)`);
    return await standardCompression(messages);
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy - EXTREME LONG CHAT EDITION', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    max_context_messages: MAX_CONTEXT_MESSAGES,
    preserve_recent_messages: PRESERVE_RECENT_MESSAGES,
    aggressive_compression: AGGRESSIVE_COMPRESSION,
    rolling_summary_size: ROLLING_SUMMARY_SIZE,
    compression_enabled: true,
    optimized_for: '500-1000+ message conversations (DeepSeek models)'
  });
});

// List models endpoint (OpenAI compatible)
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

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    const requestStart = Date.now();
    console.log(`\n📨 NEW REQUEST`);
    console.log(`   Model: ${model}`);
    console.log(`   Messages: ${messages.length}`);
    console.log(`   Estimated tokens: ~${estimateTotalTokens(messages)}`);
    
    // Apply intelligent context compression
    const compressionStart = Date.now();
    const compressedMessages = await compressContext(messages);
    const compressionTime = Date.now() - compressionStart;
    
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`⏱️  COMPRESSION TIME: ${compressionTime}ms`);
    console.log(`📤 SENDING TO API:`);
    console.log(`   Messages: ${compressedMessages.length}`);
    console.log(`   Estimated tokens: ~${estimateTotalTokens(compressedMessages)}`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'deepseek-ai/deepseek-v3.1';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'deepseek-ai/deepseek-r1-distill-qwen-14b';
        } else {
          nimModel = 'deepseek-ai/deepseek-r1-distill-qwen-7b';
        }
      }
    }
    
    console.log(`🤖 Using model: ${nimModel}`);
    
    // Transform OpenAI request to NIM format with compressed context
    const nimRequest = {
      model: nimModel,
      messages: compressedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API with extended timeout
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000 // 5 minutes timeout for long responses
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        const totalTime = Date.now() - requestStart;
        console.log(`✅ Request completed in ${totalTime}ms\n`);
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('❌ Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      const totalTime = Date.now() - requestStart;
      console.log(`✅ Request completed in ${totalTime}ms\n`);
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('\n❌ PROXY ERROR:', error.message);
    if (error.response?.data) {
      console.error('   API Response:', error.response.data);
    }
    console.log('');
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`\n╔═══════════════════════════════════════════════════════════╗`);
  console.log(`║  🚀 OpenAI to NVIDIA NIM Proxy - EXTREME LONG CHAT       ║`);
  console.log(`╚═══════════════════════════════════════════════════════════╝`);
  console.log(`\n📡 Server running on port ${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log(`\n⚙️  CONFIGURATION:`);
  console.log(`   • Max context messages: ${MAX_CONTEXT_MESSAGES}`);
  console.log(`   • Preserve recent messages: ${PRESERVE_RECENT_MESSAGES}`);
  console.log(`   • Aggressive compression: ${AGGRESSIVE_COMPRESSION ? '✅ ENABLED' : '❌ DISABLED'}`);
  console.log(`   • Rolling summary size: ${ROLLING_SUMMARY_SIZE} messages/chunk`);
  console.log(`   • Reasoning display: ${SHOW_REASONING ? '✅ ENABLED' : '❌ DISABLED'}`);
  console.log(`   • Thinking mode: ${ENABLE_THINKING_MODE ? '✅ ENABLED' : '❌ DISABLED'}`);
  console.log(`\n💪 OPTIMIZED FOR:`);
  console.log(`   • 500-1000+ message conversations`);
  console.log(`   • Fanfic planning & worldbuilding`);
  console.log(`   • DeepSeek models (unlimited & free)`);
  console.log(`   • Long-form creative writing sessions`);
  console.log(`\n🔥 Ready to handle extreme conversation lengths!\n`);
});
