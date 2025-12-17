import * as readline from 'node:readline';
import { LlamaClient } from './llama-client.js';

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  magenta: '\x1b[35m',
  red: '\x1b[31m'
};

function printWelcome(): void {
  console.log(`
${COLORS.cyan}${COLORS.bright}╔══════════════════════════════════════════╗
║       🦙 Llama Chatbot CLI               ║
╚══════════════════════════════════════════╝${COLORS.reset}

${COLORS.dim}Commands:
  ${COLORS.yellow}quit${COLORS.dim}   - Exit the chatbot
  ${COLORS.yellow}clear${COLORS.dim}  - Clear conversation history
  ${COLORS.yellow}history${COLORS.dim} - Show conversation history${COLORS.reset}
`);
}

function printUserMessage(message: string): void {
  console.log(`\n${COLORS.green}${COLORS.bright}You:${COLORS.reset} ${message}`);
}

function printAssistantMessage(message: string): void {
  console.log(`${COLORS.magenta}${COLORS.bright}Assistant:${COLORS.reset} ${message}\n`);
}

function printError(message: string): void {
  console.log(`${COLORS.red}Error:${COLORS.reset} ${message}\n`);
}

function printInfo(message: string): void {
  console.log(`${COLORS.cyan}${message}${COLORS.reset}\n`);
}

async function main(): Promise<void> {
  const client = new LlamaClient({
    baseUrl: process.env.LLAMA_URL ?? 'http://localhost:8080',
    systemPrompt: process.env.SYSTEM_PROMPT ?? 'You are a helpful, friendly assistant. Keep your responses concise but informative.',
    temperature: parseFloat(process.env.TEMPERATURE ?? '0.7'),
    maxTokens: parseInt(process.env.MAX_TOKENS ?? '2048', 10)
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  printWelcome();

  const prompt = (): void => {
    rl.question(`${COLORS.dim}>${COLORS.reset} `, async (input) => {
      const trimmed = input.trim();

      if (!trimmed) {
        prompt();
        return;
      }

      // Handle commands
      switch (trimmed.toLowerCase()) {
        case 'quit':
        case 'exit':
        case 'q':
          printInfo('Goodbye! 👋');
          rl.close();
          process.exit(0);
          break;

        case 'clear':
          client.clearHistory();
          printInfo('Conversation history cleared.');
          prompt();
          break;

        case 'history':
          const history = client.getHistory();
          console.log(`\n${COLORS.cyan}Conversation History:${COLORS.reset}`);
          history.forEach((msg, i) => {
            const roleColor = msg.role === 'user' ? COLORS.green : 
                            msg.role === 'assistant' ? COLORS.magenta : COLORS.yellow;
            console.log(`  ${COLORS.dim}${i + 1}.${COLORS.reset} ${roleColor}[${msg.role}]${COLORS.reset} ${msg.content.substring(0, 50)}${msg.content.length > 50 ? '...' : ''}`);
          });
          console.log();
          prompt();
          break;

        default:
          // Send message to LLM
          printUserMessage(trimmed);
          
          try {
            process.stdout.write(`${COLORS.magenta}${COLORS.bright}Assistant:${COLORS.reset} ${COLORS.dim}thinking...${COLORS.reset}`);
            
            const response = await client.chat(trimmed);
            
            // Clear "thinking..." and print response
            process.stdout.write('\r\x1b[K');
            printAssistantMessage(response);
          } catch (error) {
            process.stdout.write('\r\x1b[K');
            printError(error instanceof Error ? error.message : 'Unknown error occurred');
          }
          
          prompt();
          break;
      }
    });
  };

  prompt();
}

main().catch(console.error);

