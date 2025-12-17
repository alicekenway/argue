import type { Message, ChatCompletionRequest, ChatCompletionResponse, LlamaClientConfig } from './types.js';

export class LlamaClient {
  private baseUrl: string;
  private model: string;
  private systemPrompt: string;
  private temperature: number;
  private maxTokens: number;
  private history: Message[] = [];

  constructor(config: LlamaClientConfig = {}) {
    this.baseUrl = config.baseUrl ?? 'http://localhost:8080';
    this.model = config.model ?? 'default';
    this.systemPrompt = config.systemPrompt ?? 'You are a helpful assistant. Please directly talk with user, don\'t show middle thinking process';
    this.temperature = config.temperature ?? 0.7;
    this.maxTokens = config.maxTokens ?? 2048;

    // Initialize with system prompt
    this.history.push({
      role: 'system',
      content: this.systemPrompt
    });
  }

  async chat(userMessage: string): Promise<string> {
    // Add user message to history
    this.history.push({
      role: 'user',
      content: userMessage
    });

    const requestBody: ChatCompletionRequest = {
      model: this.model,
      messages: this.history,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      stream: false
    };

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error ${response.status}: ${errorText}`);
      }

      const data = await response.json() as ChatCompletionResponse;
      const assistantMessage = data.choices[0]?.message?.content ?? '';

      // Add assistant response to history
      this.history.push({
        role: 'assistant',
        content: assistantMessage
      });

      return assistantMessage;
    } catch (error) {
      // Remove the failed user message from history
      this.history.pop();
      
      if (error instanceof Error) {
        throw new Error(`Failed to chat: ${error.message}`);
      }
      throw error;
    }
  }

  clearHistory(): void {
    this.history = [{
      role: 'system',
      content: this.systemPrompt
    }];
  }

  getHistory(): Message[] {
    return [...this.history];
  }

  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
    if (this.history.length > 0 && this.history[0].role === 'system') {
      this.history[0].content = prompt;
    }
  }
}

