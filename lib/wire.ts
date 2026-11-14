import { isObject } from './providers.ts';

function textBlocks(value) {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.filter((block) => isObject(block) && block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text).join('\n');
}

function asOpenAIContent(blocks) {
  return blocks.flatMap((block) => {
    if (!isObject(block)) return [];
    if (block.type === 'text' && typeof block.text === 'string') {
      return [{ type: 'text', text: block.text }];
    }
    if (block.type === 'image' && isObject(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        const mediaType = typeof source.media_type === 'string' ? source.media_type : 'image/png';
        return [{ type: 'image_url', image_url: { url: `data:${mediaType};base64,${source.data}` } }];
      }
      if (source.type === 'url' && typeof source.url === 'string') {
        return [{ type: 'image_url', image_url: { url: source.url } }];
      }
    }
    return [];
  });
}

function toolResultContent(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    const text = textBlocks(value);
    return text || JSON.stringify(value);
  }
  return value === undefined ? '' : JSON.stringify(value);
}

function schemaObject(value) {
  return isObject(value) ? value : { type: 'object', properties: {} };
}

export function anthropicRequest(body, model) {
  const messages = [];
  const system = textBlocks(body.system);
  if (system) messages.push({ role: 'system', content: system });

  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObject(message)) continue;
    const assistant = message.role === 'assistant';
    if (typeof message.content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content: message.content });
      continue;
    }
    if (!Array.isArray(message.content)) continue;
    if (assistant) {
      const calls = message.content.filter((block) => isObject(block) && block.type === 'tool_use').map((block) => ({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(schemaObject(block.input)),
        },
      }));
      const content = textBlocks(message.content);
      messages.push({
        role: 'assistant',
        content: content || null,
        ...(calls.length ? { tool_calls: calls } : {}),
      });
      continue;
    }
    const content = asOpenAIContent(message.content);
    for (const block of message.content) {
      if (isObject(block) && block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultContent(block.content),
        });
      }
    }
    if (content.length) messages.push({ role: 'user', content });
  }

  const request = { model, messages };
  if (typeof body.max_tokens === 'number') request.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') request.temperature = body.temperature;
  if (typeof body.top_p === 'number') request.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) request.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.filter((tool) => isObject(tool) && typeof tool.name === 'string').map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: schemaObject(tool.input_schema),
      },
    }));
    if (tools.length) request.tools = tools;
  }
  if (isObject(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') request.tool_choice = 'auto';
    else if (body.tool_choice.type === 'any') request.tool_choice = 'required';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') {
      request.tool_choice = { type: 'function', function: { name: body.tool_choice.name } };
    }
  }
  if (body.stream === true) {
    request.stream = true;
    request.stream_options = { include_usage: true };
  }
  return request;
}

function anthropicContentFromOpenAI(value) {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.flatMap((part) => {
    if (!isObject(part)) return [];
    if (part.type === 'text' && typeof part.text === 'string') {
      return [{ type: 'text', text: part.text }];
    }
    if (part.type === 'image_url' && isObject(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (data) {
        return [{
          type: 'image',
          source: { type: 'base64', media_type: data[1] || 'image/png', data: data[2] },
        }];
      }
      return [{ type: 'image', source: { type: 'url', url } }];
    }
    return [];
  });
}

export function openAIRequestToAnthropic(body, model) {
  let system = '';
  const messages = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObject(message)) continue;
    if (message.role === 'system') {
      const text = textBlocks(message.content);
      if (text) system += `${system ? '\n' : ''}${text}`;
      continue;
    }
    if (message.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
          content: typeof message.content === 'string' ? message.content : JSON.stringify(message.content ?? ''),
        }],
      });
      continue;
    }
    const assistant = message.role === 'assistant';
    const source = anthropicContentFromOpenAI(message.content);
    const blocks = typeof source === 'string' ? [{ type: 'text', text: source }] : source;
    if (assistant && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isObject(call) || !isObject(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: parsedArguments(call.function.arguments),
        });
      }
    }
    messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
  }

  const request = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 512, messages };
  if (system) request.system = system;
  if (typeof body.temperature === 'number') request.temperature = body.temperature;
  if (typeof body.top_p === 'number') request.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length) request.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    request.tools = body.tools.flatMap((tool) => (
      isObject(tool) && isObject(tool.function) && typeof tool.function.name === 'string'
        ? [{
          name: tool.function.name,
          description: typeof tool.function.description === 'string' ? tool.function.description : '',
          input_schema: isObject(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
        }]
        : []
    ));
  }
  if (isObject(body.tool_choice)) {
    if (body.tool_choice.type === 'auto' || body.tool_choice.type === 'none') {
      request.tool_choice = { type: body.tool_choice.type };
    } else if (body.tool_choice.type === 'function' && isObject(body.tool_choice.function)) {
      request.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
    }
  } else if (body.tool_choice === 'required') {
    request.tool_choice = { type: 'any' };
  }
  if (body.stream === true) request.stream = true;
  return request;
}

function parsedArguments(value) {
  if (isObject(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      if (isObject(parsed)) return parsed;
    } catch {
      // Invalid function arguments have no structured input to expose.
    }
  }
  return {};
}

export function anthropicReply(body, requestedModel) {
  const choice = Array.isArray(body.choices) && isObject(body.choices[0]) ? body.choices[0] : {};
  const message = isObject(choice.message) ? choice.message : {};
  const content = [];
  if (typeof message.content === 'string' && message.content) content.push({ type: 'text', text: message.content });
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isObject(call) || !isObject(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parsedArguments(call.function.arguments),
    });
  }
  if (!content.length) content.push({ type: 'text', text: '' });
  const usage = isObject(body.usage) ? body.usage : {};
  const finish = choice.finish_reason;
  const stopReason = finish === 'length' ? 'max_tokens'
    : finish === 'tool_calls' || finish === 'function_call' ? 'tool_use'
      : finish === 'content_filter' ? 'refusal' : 'end_turn';
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model: requestedModel,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
      output_tokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
    },
  };
}

function openAIFinishReason(reason) {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

export function anthropicToOpenAIReply(body, model) {
  const content = [];
  const toolCalls = [];
  for (const block of Array.isArray(body.content) ? body.content : []) {
    if (!isObject(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') content.push(block.text);
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_tool',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isObject(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isObject(body.usage) ? body.usage : {};
  const input = Number(usage.input_tokens) || 0;
  const output = Number(usage.output_tokens) || 0;
  return {
    id: typeof body.id === 'string' ? `chatcmpl_${body.id}` : 'chatcmpl_anthropic',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: content.length ? content.join('') : null,
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: openAIFinishReason(body.stop_reason),
    }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

export function sseFrame(event, value) {
  return `${event ? `event: ${event}\n` : ''}data: ${JSON.stringify(value)}\n\n`;
}

async function* readEvents(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        yield buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield buffer;
  } finally {
    reader.releaseLock();
  }
}

function eventParts(raw) {
  let name = null;
  const data = [];
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) name = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  return { name, data: data.join('\n') };
}

export async function* anthropicEventsAsOpenAI(stream, model) {
  const makeChunk = (choices, usage) => ({
    id: 'chatcmpl_anthropic_stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices,
    ...(usage ? { usage } : {}),
  });
  const emit = (value) => `data: ${JSON.stringify(value)}\n\n`;
  let promptTokens = 0;
  let completionTokens = 0;
  let finished = false;
  const toolsByBlock = new Map();
  for await (const raw of readEvents(stream)) {
    const { name, data } = eventParts(raw);
    let payload;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isObject(payload)) continue;
    if (name === 'message_start') {
      const message = isObject(payload.message) ? payload.message : {};
      if (isObject(message.usage)) promptTokens = Number(message.usage.input_tokens) || 0;
      yield emit(makeChunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }]));
      continue;
    }
    if (name === 'content_block_start') {
      const block = isObject(payload.content_block) ? payload.content_block : {};
      if (block.type === 'tool_use') {
        const index = Number(payload.index) || 0;
        toolsByBlock.set(index, index);
        yield emit(makeChunk([{
          index: 0,
          delta: {
            tool_calls: [{
              index,
              id: typeof block.id === 'string' ? block.id : `call_tool_${index}`,
              type: 'function',
              function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
            }],
          },
          finish_reason: null,
        }]));
      }
      continue;
    }
    if (name === 'content_block_delta' && isObject(payload.delta)) {
      if (payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
        yield emit(makeChunk([{ index: 0, delta: { content: payload.delta.text }, finish_reason: null }]));
      } else if (payload.delta.type === 'input_json_delta' && typeof payload.delta.partial_json === 'string') {
        const blockIndex = Number(payload.index) || 0;
        yield emit(makeChunk([{
          index: 0,
          delta: {
            tool_calls: [{
              index: toolsByBlock.get(blockIndex) ?? blockIndex,
              function: { arguments: payload.delta.partial_json },
            }],
          },
          finish_reason: null,
        }]));
      }
      continue;
    }
    if (name === 'message_delta') {
      if (isObject(payload.usage)) completionTokens = Number(payload.usage.output_tokens) || 0;
      const delta = isObject(payload.delta) ? payload.delta : {};
      yield emit(makeChunk([{
        index: 0, delta: {}, finish_reason: openAIFinishReason(delta.stop_reason),
      }]));
      finished = true;
    }
  }
  if (!finished) yield emit(makeChunk([{ index: 0, delta: {}, finish_reason: 'stop' }]));
  yield emit(makeChunk([], {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  }));
  yield 'data: [DONE]\n\n';
}

export async function* correctAnthropicModel(stream, model) {
  for await (const raw of readEvents(stream)) {
    const { name, data } = eventParts(raw);
    if (name !== 'message_start') {
      yield `${raw}\n\n`;
      continue;
    }
    try {
      const payload = JSON.parse(data);
      if (isObject(payload) && isObject(payload.message)) payload.message.model = model;
      yield `event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
    } catch {
      yield `${raw}\n\n`;
    }
  }
}

export async function* anthropicEvents(stream, requestedModel) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let ended = false;
  let textBlock = false;
  let textIndex = -1;
  let nextBlockIndex = 0;
  const tools = new Map();
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason = null;

  const start = () => {
    if (started) return '';
    started = true;
    return sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model: requestedModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
  };

  const finish = () => {
    if (ended) return '';
    ended = true;
    let output = '';
    if (textBlock) output += sseFrame('content_block_stop', { type: 'content_block_stop', index: textIndex });
    for (const index of tools.values()) {
      output += sseFrame('content_block_stop', { type: 'content_block_stop', index });
    }
    const reason = finishReason === 'length' ? 'max_tokens'
      : finishReason === 'tool_calls' || finishReason === 'function_call' ? 'tool_use'
        : finishReason === 'content_filter' ? 'refusal' : 'end_turn';
    output += sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: reason, stop_sequence: null },
      usage: { output_tokens: outputTokens },
    });
    output += sseFrame('message_stop', { type: 'message_stop' });
    return output;
  };

  const consume = (frame) => {
    let output = '';
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') {
        output += start();
        output += finish();
        continue;
      }
      let chunk;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }
      if (!isObject(chunk)) continue;
      if (isObject(chunk.usage)) {
        inputTokens = Number(chunk.usage.prompt_tokens) || 0;
        outputTokens = Number(chunk.usage.completion_tokens) || 0;
      }
      const choice = Array.isArray(chunk.choices) && isObject(chunk.choices[0]) ? chunk.choices[0] : null;
      if (!choice) continue;
      output += start();
      const delta = isObject(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string' && delta.content) {
        if (!textBlock) {
          textBlock = true;
          textIndex = nextBlockIndex;
          nextBlockIndex += 1;
          output += sseFrame('content_block_start', {
            type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' },
          });
        }
        output += sseFrame('content_block_delta', {
          type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content },
        });
      }
      for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
        if (!isObject(call)) continue;
        const key = typeof call.index === 'number' ? call.index : tools.size;
        let index = tools.get(key);
        const fn = isObject(call.function) ? call.function : {};
        if (index === undefined) {
          if (textBlock) {
            output += sseFrame('content_block_stop', { type: 'content_block_stop', index: textIndex });
            textBlock = false;
          }
          index = nextBlockIndex;
          nextBlockIndex += 1;
          tools.set(key, index);
          output += sseFrame('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          });
        }
        if (typeof fn.arguments === 'string' && fn.arguments) {
          output += sseFrame('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          });
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) finishReason = choice.finish_reason;
    }
    return output;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let splitAt = buffer.indexOf('\n\n');
      while (splitAt >= 0) {
        const frame = buffer.slice(0, splitAt);
        buffer = buffer.slice(splitAt + 2);
        const converted = consume(frame);
        if (converted) yield converted;
        splitAt = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const converted = consume(buffer);
      if (converted) yield converted;
    }
  } finally {
    reader.releaseLock();
  }
  if (!ended) {
    const opening = start();
    if (opening) yield opening;
    const closing = finish();
    if (closing) yield closing;
  }
}

export function openAIError(message, type = 'server_error', code = null) {
  return { error: { message, type, param: null, code } };
}

export function anthropicError(type, message) {
  return { type: 'error', error: { type, message } };
}

export function openAIModels(models) {
  return {
    object: 'list',
    data: models.map((model) => ({
      id: model.id, object: 'model', created: 1767225600, owned_by: model.provider,
    })),
  };
}

export function anthropicModels(models) {
  return {
    data: models.map((model) => ({
      type: 'model', id: model.id, display_name: model.id, created_at: '2026-01-01T00:00:00Z',
    })),
    has_more: false,
  };
}
