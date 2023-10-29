import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const chatCompletion = await openAI.createChatCompletion({
  model: "gpt-3.5-turbo",
  messages: [
    { "role": "user", "content": "What is the weather like in Boston?" },
  ],
  function_call: { name: "get_current_weather" },
  functions: [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
          },
        },
        "required": ["location"],
      },
    },
  ],
});

console.log(chatCompletion);
