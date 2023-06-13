export function throwError(data: {
  error?: { type: string; message: string; code: string };
}) {
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

export async function decodeStream<T>(
  res: Response,
  callback: (data: T) => boolean
) {
  const stream = res.body!;
  const decoder = new TextDecoder();

  if (stream.locked) {
    throw new Error("The stream is locked.");
  }

  const reader = stream.getReader();
  let read = true;
  let stopReason = "Callback stop.";

  while (read) {
    const { done, value: buff } = await reader.read();

    if (done) {
      break;
    }

    try {
      const stringBuff = decoder.decode(buff);
      const chunks = stringBuff.split("\n\n");

      for (const chunk of chunks) {
        const truncatedChunk = chunk.slice(6).trim();
        if (truncatedChunk.length === 0 || truncatedChunk === "[DONE]") {
          continue;
        }

        const argument: T = JSON.parse(truncatedChunk);

        read &&= callback(argument);

        if (!read) {
          break;
        }
      }
    } catch (error) {
      stopReason = "Chunk parse error.";
      console.error(stopReason, error);
      read = false;
    }
  }

  reader.releaseLock();

  if (!read) {
    await stream.cancel(stopReason);
  }
}
