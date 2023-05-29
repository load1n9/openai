import { TextDelimiterStream } from "https://deno.land/std@0.189.0/streams/mod.ts";

export function throwError(
  data: { error?: { type: string; message: string; code: string } },
) {
  if (data.error) {
    let errorMessage = `${data.error.type}`;
    if (data.error.message) {
      errorMessage += ": " + data.error.message;
    }
    if (data.error.code) {
      errorMessage += ` (${data.error.code})`;
    }
    // console.log(data.error);
    throw new Error(errorMessage);
  }
}

// deno-lint-ignore no-explicit-any
export async function decodeStream(
  res: Response,
  callback: (data: any) => void,
) {
  const chunks = res.body!
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextDelimiterStream("\n\n"));

  for await (const chunk of chunks) {
    let data;
    try {
      data = JSON.parse(chunk);
    } catch {
      // no-op (just checking if error message)
    }
    if (data) throwError(data);

    if (chunk === "data: [DONE]") break;
    callback(JSON.parse(chunk.slice(6)));
  }
}
