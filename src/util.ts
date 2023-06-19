import { TextDelimiterStream } from "https://deno.land/std@0.189.0/streams/text_delimiter_stream.ts";
import { ModelError, RequiredError } from "./types.ts";

export function throwErrorIfNeeded(response: unknown) {
  // deno-lint-ignore ban-types
  if ("error" in (response as object)) {
    const {
      error: { type, message },
    } = response as { error: ModelError };

    throw new RequiredError(type, message);
  }
}

export async function decodeStream<T>(
  { body: stream }: Response,
  callback: (data: T) => void
) {
  if (stream === null || stream.locked) {
    throw new Error(`The stream is ${stream === null ? "null" : "locked"}.`);
  }

  const chunks = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextDelimiterStream("\n\n"))
    .getReader();

  try {
    for (;;) {
      const { done, value: chunk } = await chunks.read();

      if (done || chunk === "data: [DONE]") {
        break;
      }

      const argument = JSON.parse(
        chunk.startsWith("data: ") ? chunk.slice(6) : chunk
      );

      throwErrorIfNeeded(argument);
      callback(argument);
    }
  } finally {
    chunks.releaseLock();
  }
}
