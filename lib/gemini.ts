import { isObject } from './providers.ts';

function text(value) {
  return typeof value === 'string' ? value : '';
}

function textParts(value) {
  if (typeof value === 'string') return [{ text: value }];
  if (!Array.isArray(value)) return [];
  return value.flatMap((part) => {
    if (!isObject(part)) return [];
    if (part.type === 'text' && typeof part.text === 'string') return [{ text: part.text }];
    if (part.type === 'image_url' && isObject(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (data) return [{ inlineData: { mimeType: data[1] || 'image/png', data: data[2] } }];
      return [{ fileData: { fileUri: url, mimeType: 'image/*' } }];
    }
    return [];
  });
}

function parseJsonObject(value) {
  if (isObject(value)) return value;
  if (typeof value === 'string') {
    try {
      const result = JSON.parse(value);
      if (isObject(result)) return result;
    } catch {
      // Function arguments may be incomplete while streaming.
    }
  }
  return {};
}

function declarations(openAITools) {
  const functions = [];
  for (const tool of Array.isArray(openAITools) ? openAITools : []) {
    if (!isObject(tool) || !isObject(tool.function) || typeof tool.function.name !== 'string') continue;
    functions.push({
      name: tool.function.name,
      description: text(tool.function.description),
      parameters: isObject(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    });
  }
  return functions.length ? [{ functionDeclarations: functions }] : [];
}

function openAIHistory(messages) {
  let system = '';
  const contents = [];
  for (const message of Array.isArray(messages) ? messages : []) {
    if (!isObject(message)) continue;
    if (message.role === 'system') {
      if (typeof message.content === 'string') system += `${system ? '\n' : ''}${message.content}`;
      else system += `${system ? '\n' : ''}${message.content?.filter?.((part) => part?.type === 'text').map((part) => part.text).join('\n') ?? ''}`;
      continue;
    }
    if (message.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            name: text(message.name) || text(message.tool_call_id),
            response: { content: text(message.content) },
          },
        }],
      });
      continue;
    }
    const role = message.role === 'assistant' ? 'model' : 'user';
    const parts = textParts(message.content);
    for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
      if (!isObject(call) || !isObject(call.function)) continue;
      parts.push({
        functionCall: {
          name: text(call.function.name),
          args: parseJsonObject(call.function.arguments),
        },
      });
    }
    if (parts.length) contents.push({ role, parts });
  }
  return { system, contents };
}

export function openAIToGemini(body) {
  const { system, contents } = openAIHistory(body.messages);
  const request = { contents };
  if (system) request.systemInstruction = { parts: [{ text: system }] };
  const functions = declarations(body.tools);
  if (functions.length) request.tools = functions;

  const generationConfig = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (Array.isArray(body.stop) && body.stop.length) generationConfig.stopSequences = body.stop;
  const format = isObject(body.response_format) ? body.response_format.type : '';
  if (format === 'json_object' || format === 'json_schema') {
    generationConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(generationConfig).length) request.generationConfig = generationConfig;
  return request;
}

function anthropicParts(content, assistant) {
  if (typeof content === 'string') return [{ text: content }];
  const parts = [];
  for (const block of Array.isArray(content) ? content : []) {
    if (!isObject(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ text: block.text });
    } else if (block.type === 'image' && isObject(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: text(source.media_type) || 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url, mimeType: 'image/*' } });
      }
    } else if (assistant && block.type === 'tool_use') {
      parts.push({ functionCall: { name: text(block.name), args: isObject(block.input) ? block.input : {} } });
    } else if (!assistant && block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: text(block.name) || text(block.tool_use_id),
          response: { content: typeof block.content === 'string' ? block.content : JSON.stringify(block.content ?? '') },
        },
      });
    }
  }
  return parts;
}

export function anthropicToGemini(body) {
  const request = { contents: [] };
  if (typeof body.system === 'string' && body.system) {
    request.systemInstruction = { parts: [{ text: body.system }] };
  } else if (Array.isArray(body.system)) {
    const systemParts = body.system.flatMap((part) => (
      isObject(part) && part.type === 'text' && typeof part.text === 'string' ? [{ text: part.text }] : []
    ));
    if (systemParts.length) request.systemInstruction = { parts: systemParts };
  }
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObject(message)) continue;
    const assistant = message.role === 'assistant';
    const parts = anthropicParts(message.content, assistant);
    if (parts.length) request.contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const tools = Array.isArray(body.tools) ? body.tools.flatMap((tool) => (
    isObject(tool) && typeof tool.name === 'string'
      ? [{
        name: tool.name,
        description: text(tool.description),
        parameters: isObject(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      }]
      : []
  )) : [];
  if (tools.length) request.tools = [{ functionDeclarations: tools }];

  const generationConfig = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) {
    generationConfig.stopSequences = body.stop_sequences;
  }
  if (Object.keys(generationConfig).length) request.generationConfig = generationConfig;
  return request;
}

function finishReason(reason) {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'content_filter';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

function stopReason(reason) {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'refusal';
  if (reason === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

function answerData(body) {
  const candidate = Array.isArray(body.candidates) && isObject(body.candidates[0]) ? body.candidates[0] : {};
  const content = isObject(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts = [];
  const calls = [];
  for (const part of parts) {
    if (!isObject(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isObject(part.functionCall) && typeof part.functionCall.name === 'string') {
      const index = calls.length;
      calls.push({
        id: `call_${part.functionCall.name}_${index}`,
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(isObject(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  const usage = isObject(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    toolCalls: calls,
    finish: finishReason(candidate.finishReason),
    stop: stopReason(candidate.finishReason),
    usage: {
      input: Number(usage.promptTokenCount) || 0,
      output: Number(usage.candidatesTokenCount) || 0,
      total: Number(usage.totalTokenCount) || 0,
    },
  };
}

export function geminiToOpenAI(body, model, id = 'chatcmpl-gemini') {
  const answer = answerData(body);
  return {
    id,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: answer.text || null,
        ...(answer.toolCalls.length ? { tool_calls: answer.toolCalls } : {}),
      },
      finish_reason: answer.finish,
    }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.total,
    },
  };
}

export function geminiToAnthropic(body, model) {
  const answer = answerData(body);
  const content = [];
  if (answer.text) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id,
      name: call.function.name,
      input: parseJsonObject(call.function.arguments),
    });
  }
  if (!content.length) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: answer.stop,
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

export function geminiStreamUrl(base, model) {
  return `${base}/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
}

export async function* geminiToOpenAIEvents(stream, model) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sawRole = false;
  let finished = false;
  let usage = null;
  let callIndex = 0;
  const toolCalls = new Map();
  const mergeArguments = (target, source) => {
    for (const [key, value] of Object.entries(source)) {
      if (isObject(value) && isObject(target[key])) mergeArguments(target[key], value);
      else target[key] = value;
    }
    return target;
  };
  const event = (chunk) => `data: ${JSON.stringify(chunk)}\n\n`;
  const baseChunk = (choices, extra = {}) => ({
    id: 'chatcmpl-gemini-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices,
    ...extra,
  });
  const flushToolCalls = () => {
    if (!toolCalls.size) return '';
    const calls = [...toolCalls.values()].map((call) => ({
      index: call.index,
      id: call.id,
      type: 'function',
      function: {
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      },
    }));
    toolCalls.clear();
    return event(baseChunk([{
      index: 0,
      delta: { tool_calls: calls },
      finish_reason: null,
    }]));
  };

  const convert = (frame) => {
    const output = [];
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      let chunk;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isObject(chunk)) continue;
      const candidate = Array.isArray(chunk.candidates) && isObject(chunk.candidates[0]) ? chunk.candidates[0] : {};
      const parts = isObject(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
      if (!sawRole) {
        sawRole = true;
        output.push(event(baseChunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }])));
      }
      for (const [partIndex, part] of parts.entries()) {
        if (!isObject(part)) continue;
        if (typeof part.text === 'string' && part.text) {
          output.push(event(baseChunk([{ index: 0, delta: { content: part.text }, finish_reason: null }])));
        }
        if (isObject(part.functionCall) && typeof part.functionCall.name === 'string') {
          const name = part.functionCall.name;
          const key = `${partIndex}:${name}`;
          let call = toolCalls.get(key);
          if (!call) {
            const index = callIndex++;
            call = { index, id: `call_${name}_${index}`, name, arguments: {} };
            toolCalls.set(key, call);
          }
          if (isObject(part.functionCall.args)) {
            mergeArguments(call.arguments, part.functionCall.args);
          }
        }
      }
      if (isObject(chunk.usageMetadata)) {
        usage = {
          prompt_tokens: Number(chunk.usageMetadata.promptTokenCount) || 0,
          completion_tokens: Number(chunk.usageMetadata.candidatesTokenCount) || 0,
          total_tokens: Number(chunk.usageMetadata.totalTokenCount) || 0,
        };
      }
      if (candidate.finishReason) {
        const toolCallEvent = flushToolCalls();
        if (toolCallEvent) output.push(toolCallEvent);
        output.push(event(baseChunk([{
          index: 0, delta: {}, finish_reason: finishReason(candidate.finishReason),
        }])));
        finished = true;
      }
    }
    return output;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const frames = convert(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        for (const frame of frames) yield frame;
        boundary = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) for (const frame of convert(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
  if (!sawRole) yield event(baseChunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }]));
  const toolCallEvent = flushToolCalls();
  if (toolCallEvent) yield toolCallEvent;
  if (!finished) yield event(baseChunk([{ index: 0, delta: {}, finish_reason: 'stop' }]));
  if (usage) yield event(baseChunk([], { usage }));
  yield 'data: [DONE]\n\n';
}

export async function* geminiToAnthropicEvents(stream, model) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let textOpen = false;
  let textIndex = -1;
  let nextIndex = 0;
  let usage = { input_tokens: 0, output_tokens: 0 };
  let finish = 'STOP';
  const toolCalls = new Map();
  const mergeArguments = (target, source) => {
    for (const [key, value] of Object.entries(source)) {
      if (isObject(value) && isObject(target[key])) mergeArguments(target[key], value);
      else target[key] = value;
    }
    return target;
  };
  const frame = (type, value) => `event: ${type}\ndata: ${JSON.stringify(value)}\n\n`;
  const startMessage = () => {
    if (started) return '';
    started = true;
    return frame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_gemini_stream', type: 'message', role: 'assistant', model,
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
      },
    });
  };
  const closeMessage = () => {
    if (finished) return '';
    finished = true;
    let output = '';
    if (textOpen) output += frame('content_block_stop', { type: 'content_block_stop', index: textIndex });
    output += frame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason(finish), stop_sequence: null },
      usage: { output_tokens: usage.output_tokens },
    });
    output += frame('message_stop', { type: 'message_stop' });
    return output;
  };
  const flushToolCalls = () => {
    let output = '';
    if (textOpen) {
      output += frame('content_block_stop', { type: 'content_block_stop', index: textIndex });
      textOpen = false;
    }
    for (const call of toolCalls.values()) {
      const index = nextIndex++;
      output += frame('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id,
          name: call.name,
          input: {},
        },
      });
      output += frame('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.arguments) },
      });
      output += frame('content_block_stop', { type: 'content_block_stop', index });
    }
    toolCalls.clear();
    return output;
  };
  const convert = (rawFrame) => {
    let output = '';
    for (const line of rawFrame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      let chunk;
      try {
        chunk = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      if (!isObject(chunk)) continue;
      const candidate = Array.isArray(chunk.candidates) && isObject(chunk.candidates[0]) ? chunk.candidates[0] : {};
      const parts = isObject(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
      if (isObject(chunk.usageMetadata)) {
        usage = {
          input_tokens: Number(chunk.usageMetadata.promptTokenCount) || 0,
          output_tokens: Number(chunk.usageMetadata.candidatesTokenCount) || 0,
        };
      }
      output += startMessage();
      for (const [partIndex, part] of parts.entries()) {
        if (!isObject(part)) continue;
        if (typeof part.text === 'string' && part.text) {
          if (!textOpen) {
            textOpen = true;
            textIndex = nextIndex++;
            output += frame('content_block_start', {
              type: 'content_block_start', index: textIndex,
              content_block: { type: 'text', text: '' },
            });
          }
          output += frame('content_block_delta', {
            type: 'content_block_delta', index: textIndex,
            delta: { type: 'text_delta', text: part.text },
          });
        }
        if (isObject(part.functionCall) && typeof part.functionCall.name === 'string') {
          const name = part.functionCall.name;
          const key = `${partIndex}:${name}`;
          let call = toolCalls.get(key);
          if (!call) {
            const index = toolCalls.size;
            call = { id: `call_${name}_${index}`, name, arguments: {} };
            toolCalls.set(key, call);
          }
          if (isObject(part.functionCall.args)) {
            mergeArguments(call.arguments, part.functionCall.args);
          }
        }
      }
      if (candidate.finishReason) {
        finish = candidate.finishReason;
        output += flushToolCalls();
        output += closeMessage();
      }
    }
    return output;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const converted = convert(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (converted) yield converted;
        boundary = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const converted = convert(buffer);
      if (converted) yield converted;
    }
  } finally {
    reader.releaseLock();
  }
  const opened = startMessage();
  if (opened) yield opened;
  const toolCallEvents = flushToolCalls();
  if (toolCallEvents) yield toolCallEvents;
  const closed = closeMessage();
  if (closed) yield closed;
}
