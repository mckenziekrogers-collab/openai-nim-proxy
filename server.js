// server.js - OpenAI to NVIDIA NIM API Proxy (Optimized for Long Responses)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 CONFIGURATION - Adjust these for different behaviors
const SHOW_REASONING = false; // Set to true to show model's thinking process
const ENABLE_THINKING_MODE = false; // Set to true for thinking-capable models

// Model mapping - Maps OpenAI model names to NVIDIA NIM models
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-70b-instruct',
  'gpt-4': 'meta/llama-3.3-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'meta/llama-3.1-405b-instruct',
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    version: '2.0-optimized'
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

// Chat completions endpoint (main proxy) - OPTIMIZED FOR LONGER RESPONSES
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Validate required fields
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'messages field is required and must be a non-empty array',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Set default model if not provided
    const requestModel = model || 'gpt-4o';
    console.log('📨 Requested model:', requestModel);
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[requestModel];
    
    if (!nimModel) {
      // Try to use the model directly with NVIDIA API
      try {
        const testResponse = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: requestModel,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 
            'Authorization': `Bearer ${NIM_API_KEY}`, 
            'Content-Type': 'application/json' 
          },
          validateStatus: (status) => status < 500
        });
        
        if (testResponse.status >= 200 && testResponse.status < 300) {
          nimModel = requestModel;
          console.log('✅ Model accepted directly:', requestModel);
        }
      } catch (e) {
        console.log('⚠️  Model test failed, using fallback');
      }
      
      // Intelligent fallback to similar models
      if (!nimModel) {
        const modelLower = requestModel.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
        console.log('🔄 Fallback to:', nimModel);
      }
    }
    
    console.log('✅ Final NIM model:', nimModel);
    
    // 🔥 ENHANCEMENT: Add instruction to encourage detailed responses
    const enhancedMessages = JSON.parse(JSON.stringify(messages)); // Deep clone
    
    // Find the last user message and enhance it for longer responses
    for (let i = enhancedMessages.length - 1; i >= 0; i--) {
      if (enhancedMessages[i].role === 'user') {
        // Append instruction (invisible to user in most clients)
        enhancedMessages[i].content += '\n\n[Provide a detailed, comprehensive response. Be thorough and elaborate on all points.]';
        break;
      }
    }
    
    // Transform OpenAI request to NIM format with optimized settings
    const nimRequest = {
      model: nimModel,
      messages: enhancedMessages,
      temperature: temperature || 0.85,  // Higher temperature = more creative/verbose
      max_tokens: Math.max(max_tokens || 0, 16000),  // Force minimum 16K tokens
      top_p: 0.95,  // Nucleus sampling for diversity
      frequency_penalty: -0.3,  // Negative value encourages more tokens
      presence_penalty: 0.0,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    console.log('📤 Request config:', {
      model: nimModel,
      max_tokens: nimRequest.max_tokens,
      temperature: nimRequest.temperature,
      top_p: nimRequest.top_p,
      frequency_penalty: nimRequest.frequency_penalty,
      message_count: enhancedMessages.length
    });
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 120000  // 2 minute timeout for long responses
    });
    
    if (stream) {
      // Handle streaming response with reasoning support
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      let totalChunks = 0;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              console.log(`📥 Stream complete: ${totalChunks} chunks sent`);
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                totalChunks++;
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
                  // Hide reasoning, only show final content
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
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('❌ Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: requestModel,
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
      
      console.log('📥 Response stats:', {
        content_length: openaiResponse.choices[0]?.message?.content?.length || 0,
        completion_tokens: openaiResponse.usage.completion_tokens,
        finish_reason: openaiResponse.choices[0]?.finish_reason,
        characters: openaiResponse.choices[0]?.message?.content?.length || 0,
        words: (openaiResponse.choices[0]?.message?.content?.split(/\s+/).length || 0)
      });
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    if (error.response?.data) {
      console.error('Error details:', JSON.stringify(error.response.data, null, 2));
    }
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500,
        details: error.response?.data || null
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Available endpoints: /health, /v1/models, /v1/chat/completions`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log(`🚀 OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/health`);
  console.log(`🔧 Configuration:`);
  console.log(`   - Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   - Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   - Max tokens: 16000 (optimized for long responses)`);
  console.log(`   - Temperature: 0.85 (optimized for verbosity)`);
  console.log(`   - Frequency penalty: -0.3 (encourages longer output)`);
  console.log('='.repeat(60));
});
