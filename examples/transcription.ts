import { OpenAI } from "../mod.ts";

const openAI = new OpenAI("YOUR_API_KEY");

const transcription = await openAI.createTranscription({
  model: "whisper-1",
  file: "./testdata/jfk.wav", // TODO: Do this more portably
});

console.log(transcription);
