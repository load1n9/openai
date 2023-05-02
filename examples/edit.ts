import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const edit = await openAI.createEdit({
  model: "text-davinci-edit-001",
  input: "What day of the wek is it?",
  instruction: "Fix the spelling mistakes",
});

console.log(edit);
