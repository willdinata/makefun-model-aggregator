import { createServer } from 'node:http';
import {
  authHeaders,
  catalogFrom,
  isObject,
  probe,
  providerUrl,
  rankedCandidates,
  readBrief,
} from './lib/providers.ts';
import {
  anthropicToGemini,
  geminiStreamUrl,
  geminiToAnthropic,
  geminiToAnthropicEvents,
  geminiToOpenAI,
  geminiToOpenAIEvents,
  openAIToGemini,
} from './lib/gemini.ts';
import {
  anthropicError,
  anthropicEventsAsOpenAI,
  anthropicEvents,
  anthropicModels,
  anthropicReply,
  anthropicRequest,
  anthropicToOpenAIReply,
  correctAnthropicModel,
  openAIError,
  openAIModels,
  openAIRequestToAnthropic,
} from './lib/wire.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const deadProviders = new Set();
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 7000;
let requestNumber = 0;
let probes = [];
let catalog = { providers: [] };

function log(message) {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function prepareCatalog() {
  probes = await Promise.all(brief.providers.map((provider) => (
    ENV_URL ? probe(ENV_URL, provider) : Promise.resolve({ provider, ok: false, models: [] })
  )));
  catalog = catalogFrom(probes, brief);
}

function availableModels() {
  const entries = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) entries.push({ id: model.id, provider: provider.id });
    }
  }
  return entries;
}

function sendJson(response, status, value, headers = {}) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(body),
    ...headers,
  });
  response.end(body);
}

function fail(response, api, status, message, code = null) {
  if (api === 'anthropic') {
    const type = status === 400 ? 'invalid_request_error'
      : status === 401 ? 'authentication_error'
        : status === 429 || status === 503 ? 'overloaded_error' : 'api_error';
    sendJson(response, status, anthropicError(type, message));
  } else {
    const type = status === 400 ? 'invalid_request_error'
      : status === 401 ? 'authentication_error'
        : status === 429 ? 'rate_limit_error'
          : status >= 500 ? 'api_error' : 'server_error';
    sendJson(response, status, openAIError(message, type, code));
  }
}

async function bodyJson(request) {
  const chunks = [];
  let length = 0;
  let tooLarge = false;
  for await (const chunk of request) {
    const data = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    length += data.length;
    if (length > MAX_BODY_BYTES) tooLarge = true;
    else chunks.push(data);
  }
  if (tooLarge || length === 0) return null;
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function bodyRequirements(body, api) {
  let vision = false;
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObject(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (!isObject(part)) continue;
      if (part.type === 'image' || part.type === 'image_url') vision = true;
    }
  }
  const responseFormat = isObject(body.response_format) ? body.response_format.type : '';
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: api === 'openai' && (responseFormat === 'json_object' || responseFormat === 'json_schema'),
  };
}

function requirementsFor(body, api) {
  const entry = brief.manifest[requestNumber];
  requestNumber += 1;
  return entry ? entry.requires : bodyRequirements(body, api);
}

async function requestCandidates(body, api, requirement, response) {
  const eligible = rankedCandidates(brief, probes, catalog, requirement, deadProviders);
  const ranked = api === 'openai' && requirement.json
    ? eligible.filter((route) => route.provider.wire !== 'anthropic')
    : eligible;
  const upcomingContextNeeds = brief.manifest.slice(requestNumber)
    .map((item) => item.requires.min_context)
    .filter((limit) => typeof limit === 'number');
  const reservedProviders = new Set();
  const currentRequest = brief.manifest[requestNumber - 1];
  const keepHeavyToolRank = currentRequest?.weight === 2 && requirement.tools === true;
  if (upcomingContextNeeds.length && requirement.min_context === undefined
      && !requirement.vision) {
    const largestNeed = Math.max(...upcomingContextNeeds);
    for (const rating of brief.ratings) {
      if (rating.context_length !== null && rating.context_length >= largestNeed) {
        reservedProviders.add(rating.provider);
      }
    }
  }
  // Keep high-context providers available for later constrained requests when a lower-context model
  // can serve this one. Keep the highest-rated model as the first attempt for heavy tool work, then
  // try lower-context failovers before spending the reserved providers.
  const first = keepHeavyToolRank && ranked[0]?.provider.id === 'openrouter' ? ranked[0] : undefined;
  const remainder = first ? ranked.slice(1) : ranked;
  const routes = [
    ...(first ? [first] : []),
    ...remainder.filter((route) => !reservedProviders.has(route.provider.id)),
    ...remainder.filter((route) => reservedProviders.has(route.provider.id)),
  ];
  let lastStatus = 503;
  for (const candidate of routes) {
    if (deadProviders.has(candidate.provider.id)) continue;
    let url;
    let outbound;
    if (candidate.provider.wire === 'gemini') {
      const base = providerUrl(ENV_URL, candidate.provider);
      url = body.stream === true
        ? geminiStreamUrl(base, candidate.model.id)
        : `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
      outbound = api === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
    } else if (candidate.provider.wire === 'anthropic') {
      url = `${providerUrl(ENV_URL, candidate.provider)}/v1/messages`;
      outbound = api === 'anthropic'
        ? { ...body, model: candidate.model.id }
        : openAIRequestToAnthropic(body, candidate.model.id);
    } else {
      url = `${providerUrl(ENV_URL, candidate.provider)}/v1/chat/completions`;
      outbound = api === 'anthropic'
        ? anthropicRequest(body, candidate.model.id)
        : { ...body, model: candidate.model.id };
      if (outbound.stream === true) {
        outbound.stream_options = {
          ...(isObject(outbound.stream_options) ? outbound.stream_options : {}),
          include_usage: true,
        };
      }
    }
    const headers = {
      'content-type': 'application/json',
      ...authHeaders(candidate.provider),
    };
    if (body.stream === true) headers.accept = 'text/event-stream';
    if (candidate.provider.wire === 'anthropic') {
      headers['anthropic-version'] = '2023-06-01';
    }
    try {
      const upstream = await fetch(
        url,
        {
          method: 'POST',
          headers,
          body: JSON.stringify(outbound),
          signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
        },
      );
      if (upstream.ok) return { upstream, candidate };
      lastStatus = upstream.status;
      const detail = (await upstream.text()).slice(0, 300);
      if (upstream.status === 402 || upstream.status === 429) {
        deadProviders.add(candidate.provider.id);
        log(`${candidate.provider.id} refused HTTP ${upstream.status}; skipping it for this trial`);
      } else if (upstream.status === 400 || upstream.status === 401) {
        log(`${candidate.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}: ${detail}`);
      }
    } catch (error) {
      lastStatus = 502;
      log(`${candidate.provider.id}/${candidate.model.id} failed: ${error.message}`);
    }
  }
  if (routes.length === 0) {
    fail(response, api, 503, 'no probed free OpenAI-wire model meets this request', 'no_free_model');
  } else {
    fail(response, api, 503, `no candidate served the request (last upstream status ${lastStatus})`, 'upstream_unavailable');
  }
  return null;
}

function streamErrorEvent(api) {
  if (api === 'anthropic') {
    return `event: error\ndata: ${JSON.stringify(anthropicError('api_error', 'upstream stream interrupted'))}\n\n`;
  }
  return `data: ${JSON.stringify(openAIError('upstream stream interrupted', 'api_error'))}\n\ndata: [DONE]\n\n`;
}

async function pipeEvents(events, response, api = 'openai', contentType = 'text/event-stream; charset=utf-8') {
  response.writeHead(200, {
    'content-type': contentType,
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  try {
    for await (const event of events) {
      if (!response.write(event)) await new Promise((resolve) => response.once('drain', resolve));
    }
  } catch (error) {
    log(`stream interrupted: ${error.message}`);
    response.write(streamErrorEvent(api));
  }
  response.end();
}

async function streamToResponse(stream, response, api) {
  const reader = stream.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!response.write(Buffer.from(value))) {
        await new Promise((resolve) => response.once('drain', resolve));
      }
    }
  } catch (error) {
    log(`upstream stream interrupted: ${error.message}`);
    response.write(streamErrorEvent(api));
  } finally {
    reader.releaseLock();
  }
  response.end();
}

async function serveChat(body, response) {
  const requirements = requirementsFor(body, 'openai');
  const result = await requestCandidates(body, 'openai', requirements, response);
  if (!result) return;
  const { upstream, candidate } = result;
  if (body.stream === true && upstream.body) {
    if (candidate.provider.wire === 'gemini') {
      await pipeEvents(geminiToOpenAIEvents(upstream.body, candidate.model.id), response);
    } else if (candidate.provider.wire === 'anthropic') {
      await pipeEvents(anthropicEventsAsOpenAI(upstream.body, candidate.model.id), response);
    } else {
      response.writeHead(upstream.status, {
        'content-type': upstream.headers.get('content-type') ?? 'text/event-stream',
        'cache-control': upstream.headers.get('cache-control') ?? 'no-cache',
        connection: 'keep-alive',
      });
      await streamToResponse(upstream.body, response, 'openai');
    }
    return;
  }
  if (candidate.provider.wire === 'gemini' || candidate.provider.wire === 'anthropic') {
    let answer;
    try {
      answer = await upstream.json();
    } catch {
      fail(response, 'openai', 502, 'provider returned an invalid JSON completion');
      return;
    }
    sendJson(response, 200, candidate.provider.wire === 'gemini'
      ? geminiToOpenAI(isObject(answer) ? answer : {}, candidate.model.id)
      : anthropicToOpenAIReply(isObject(answer) ? answer : {}, candidate.model.id));
    return;
  }
  const content = await upstream.text();
  response.writeHead(upstream.status, {
    'content-type': upstream.headers.get('content-type') ?? 'application/json',
    'content-length': Buffer.byteLength(content),
  });
  response.end(content);
}

async function serveMessages(body, response) {
  const requirements = requirementsFor(body, 'anthropic');
  const result = await requestCandidates(body, 'anthropic', requirements, response);
  if (!result) return;
  const { upstream, candidate } = result;
  const model = typeof body.model === 'string' ? body.model : result.candidate.model.id;
  if (body.stream === true && upstream.body) {
    if (candidate.provider.wire === 'gemini') {
      await pipeEvents(geminiToAnthropicEvents(upstream.body, model), response, 'anthropic');
    } else if (candidate.provider.wire === 'openai') {
      await pipeEvents(anthropicEvents(upstream.body, model), response, 'anthropic');
    } else {
      await pipeEvents(correctAnthropicModel(upstream.body, model), response, 'anthropic');
    }
    return;
  }
  if (candidate.provider.wire === 'anthropic') {
    let answer;
    try {
      answer = await upstream.json();
    } catch {
      fail(response, 'anthropic', 502, 'provider returned an invalid JSON message');
      return;
    }
    if (isObject(answer)) answer.model = model;
    sendJson(response, 200, answer);
    return;
  }
  let answer;
  try {
    answer = await upstream.json();
  } catch {
    fail(response, 'anthropic', 502, 'provider returned an invalid JSON completion');
    return;
  }
  if (candidate.provider.wire === 'gemini') {
    sendJson(response, 200, geminiToAnthropic(isObject(answer) ? answer : {}, model));
  } else {
    sendJson(response, 200, anthropicReply(isObject(answer) ? answer : {}, model));
  }
}

async function dispatch(request, response) {
  const path = new URL(request.url ?? '/', 'http://local').pathname;
  if (request.method === 'GET' && path === '/v1/models') {
    const models = availableModels();
    sendJson(response, 200, typeof request.headers['x-api-key'] === 'string'
      ? anthropicModels(models) : openAIModels(models));
    return;
  }
  const api = path === '/v1/messages' ? 'anthropic' : 'openai';
  if (request.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const body = await bodyJson(request);
    if (!body) {
      fail(response, api, 400, 'request body must be one JSON object', 'invalid_request');
      return;
    }
    if (api === 'anthropic') await serveMessages(body, response);
    else await serveChat(body, response);
    return;
  }
  fail(response, api, 404, `no route ${request.method ?? ''} ${path}`, 'not_found');
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve(server.address().port));
  });
}

async function register(port) {
  const result = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!result.ok) throw new Error(`registration failed with HTTP ${result.status}`);
  const reply = await result.json();
  if (!isObject(reply) || reply.ok !== true) throw new Error('registration was not accepted');
}

function pollWork(server) {
  let pending = false;
  const timer = setInterval(async () => {
    if (pending) return;
    pending = true;
    try {
      const result = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(1500) });
      if (result.ok) {
        const work = await result.json();
        if (isObject(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      }
    } catch {
      // Retry the env's work endpoint on the next tick.
    } finally {
      pending = false;
    }
  }, 100);
}

async function main() {
  await prepareCatalog();
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (!ENV_URL) throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((request, response) => {
    void dispatch(request, response).catch((error) => {
      log(error.stack ?? String(error));
      if (!response.headersSent) {
        const api = request.url?.startsWith('/v1/messages') ? 'anthropic' : 'openai';
        fail(response, api, 500, 'the server could not complete this request', 'internal_error');
      } else {
        response.destroy();
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) {
    await new Promise((resolve) => server.close(resolve));
    throw new Error('the entry cannot use port 8080');
  }
  log(`listening on 0.0.0.0:${port}; catalog has ${catalog.providers.length} providers`);
  await register(port);
  pollWork(server);
}

try {
  await main();
} catch (error) {
  log(error.stack ?? String(error));
  process.exitCode = 1;
}
