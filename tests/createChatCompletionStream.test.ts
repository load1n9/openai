import { assertEquals } from "https://deno.land/std@0.192.0/testing/asserts.ts";
import { OpenAI } from "../mod.ts";

const permissions = {
  net: true,
  env: true,
} satisfies Deno.PermissionOptions;

const openai = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

Deno.test(
  "createChatCompletionStream",
  { permissions },
  async () =>
    await openai.createChatCompletionStream(
      {
        model: "gpt-3.5-turbo",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Who won the world series in 2020?" },
          {
            role: "assistant",
            content: "The Los Angeles Dodgers won the World Series in 2020.",
          },
          { role: "user", content: "Where was it played?" },
        ],
      },
      (chunk) => {
        assertEquals(chunk.object, "chat.completion.chunk");
      }
    )
);
