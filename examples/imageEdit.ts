import { OpenAI } from "../mod.ts";

const openAI = new OpenAI(Deno.env.get("YOUR_API_KEY")!);

const imageEdit = await openAI.createImageEdit({
  image: "@otter.png",
  mask: "@mask.png",
  prompt: "A cute baby sea otter wearing a beret",
  n: 2,
  size: "1024x1024",
});

console.log(imageEdit);
