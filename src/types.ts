export type FileSpecifier = string | File;

export interface CompletionOptions {
  /**
   * ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-model
   */
  model: string;

  /**
   * The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
   * Note that <|endoftext|> is the document separator that the model sees during training,
   * so if a prompt is not specified the model will generate as if from the beginning of a new document.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-prompt
   */
  prompt: string | string[];

  /**
   * The suffix that comes after a completion of inserted text.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-suffix
   */
  suffix?: string;

  /**
   * The maximum number of tokens to generate in the completion.
   * The token count of your prompt plus max_tokens cannot exceed the model's context length.
   * Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
   */
  maxTokens?: number;

  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
   * while lower values like 0.2 will make it more focused and deterministic.
   * We generally recommend altering this or top_p but not both.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
   */
  temperature?: number;

  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
   * So 0.1 means only the tokens comprising the top 10% probability mass are considered.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-top_p
   */
  topP?: number;

  /**
   * How many completions to generate for each prompt.
   * Note: Because this parameter generates many completions, it can quickly consume your token quota.
   * Use carefully and ensure that you have reasonable settings for max_tokens and stop.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-n
   */
  n?: number;

  /**
   * Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
   * For example, if logprobs is 5, the API will return a list of the 5 most likely tokens.
   * The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
   * The maximum value for logprobs is 5. If you need more than this, please contact us through our Help center and describe your use case.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
   */
  logprobs?: number;

  /**
   * Echo back the prompt in addition to the completion
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
   */
  echo?: boolean;

  /**
   * Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-stop
   */
  stop?: string | string[];

  /**
   * Number between -2.0 and 2.0.
   * Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-presence_penalty
   */
  presencePenalty?: number;

  /**
   * Number between -2.0 and 2.0.
   * Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-frequency_penalty
   */
  frequencyPenalty?: number;

  /**
   * Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
   * When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.
   * Note: Because this parameter generates many completions, it can quickly consume your token quota.
   * Use carefully and ensure that you have reasonable settings for max_tokens and stop.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-best_of
   */
  bestOf?: number;

  /**
   * Modify the likelihood of specified tokens appearing in the completion.
   * Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100.
   * You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs.
   * Mathematically, the bias is added to the logits generated by the model prior to sampling.
   * The exact effect will vary per model,
   * but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
   * As an example, you can pass {"50256": -100} to prevent the <|endoftext|> token from being generated.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
   */
  logitBias?: Record<string, number>;

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/completions/create#completions/create-user
   */
  user?: string;
}

export interface ChatCompletionOptions {
  /**
   * ID of the model to use. Currently gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301 are supported.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-model
   */
  model: string;

  /**
   * The messages to generate chat completions for, in the chat format.The messages to generate chat completions for, in the chat format.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-messages
   */
  messages: ChatCompletionMessage[];

  /**
   * What sampling temperature to use, between 0 and 2.
   * Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
   * We generally recommend altering this or top_p but not both.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-temperature
   */
  temperature?: number;

  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
   * So 0.1 means only the tokens comprising the top 10% probability mass are considered.
   * We generally recommend altering this or temperature but not both.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-top_p
   */
  topP?: number;

  /**
   * How many chat completion choices to generate for each input message.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-n
   */
  n?: number;

  /**
   * Up to 4 sequences where the API will stop generating further tokens.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop
   */
  stop?: string | string[];

  /**
   * The maximum number of tokens allowed for the generated answer.
   * By default, the number of tokens the model can return will be (4096 - prompt tokens).
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens
   */
  maxTokens?: number;

  /**
   * Number between -2.0 and 2.0.
   * Positive values penalize new tokens based on whether they appear in the text so far,
   * increasing the model's likelihood to talk about new topics.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-presence_penalty
   */
  presencePenalty?: number;

  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
   * decreasing the model's likelihood to repeat the same line verbatim.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-frequency_penalty
   */
  frequencyPenalty?: number;

  /**
   * Modify the likelihood of specified tokens appearing in the completion.
   * Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
   * Mathematically, the bias is added to the logits generated by the model prior to sampling.
   * The exact effect will vary per model,
   * but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-logit_bias
   */
  logitBias?: Record<string, number>;

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-user
   */
  user?: string;

  /**
   * A list of functions the model may generate JSON inputs for.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
   */
  functions?: ChatCompletionOptionsFunction[];

  /**
   * Controls how the model responds to function calls.
   * "none" means the model does not call a function, and responds to the end-user.
   * "auto" means the model can pick between an end-user or calling a function.
   *  Specifying a particular function via {"name":\ "my_function"} forces the model to call that function.
   *  "none" is the default when no functions are present. "auto" is the default if functions are present.
   * https://platform.openai.com/docs/api-reference/chat/create#chat/create-function_call
   */
  function_call?: "none" | "auto" | { name: string };
}

export type ChatCompletionOptionsFunction = {
  name: string;
  description: string;
  parameters: ObjectSchema;
};

export interface SystemCompletionMessage {
  content: string;
  name?: string;
  role: "system";
}

export interface UserCompletionMessage {
  content: string;
  name?: string;
  role: "user";
}

export interface AssistantCompletionMessage {
  content: string;
  name?: string;
  role: "assistant";
}

export interface FunctionAwareAssistantCompletionMessage {
  content: string | null;
  role: "assistant";
  function_call?: {
    "name": string;
    "arguments": string;
  };
}

export interface FunctionCompletionMessage {
  content: string;
  role: "function";
  name: string;
}

export type ChatCompletionMessage =
  | SystemCompletionMessage
  | UserCompletionMessage
  | FunctionAwareAssistantCompletionMessage
  | FunctionCompletionMessage
  | AssistantCompletionMessage;

type JSONSchema =
  & (
    | ObjectSchema
    | StringSchema
    | NumberSchema
    | BooleanSchema
    | ArraySchema
  )
  & { description?: string };

type ObjectSchema = {
  type: "object";
  properties: Record<string, JSONSchema>;
  required: string[];
};

type ArraySchema = {
  type: "array";
  items: JSONSchema;
};

type StringSchema = {
  type: "string";
  enum?: string[];
};

type NumberSchema = {
  type: "number";
  minimum?: number;
  maximum?: number;
};

type BooleanSchema = {
  type: "boolean";
};

export interface EditOptions {
  /**
   * ID of the model to use. You can use the text-davinci-edit-001 or code-davinci-edit-001 model with this endpoint.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-model
   */
  model: string;

  /**
   * The input text to use as a starting point for the edit.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-input
   */
  input?: string;

  /**
   * The instruction that tells the model how to edit the prompt.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-instruction
   */
  instruction: string;

  /**
   * How many edits to generate for the input and instruction.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-n
   */
  n?: number;

  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
   * while lower values like 0.2 will make it more focused and deterministic.
   * We generally recommend altering this or top_p but not both.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-temperature
   */
  temperature?: number;

  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
   * So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.
   * https://platform.openai.com/docs/api-reference/edits/create#edits/create-top_p
   */
  topP?: number;
}

export interface ImageOptions {
  /**
   * A text description of the desired image(s). The maximum length is 1000 characters.
   * https://platform.openai.com/docs/api-reference/images/create#images/create-prompt
   */
  prompt: string;

  /**
   * The number of images to generate. Must be between 1 and 10.
   * https://platform.openai.com/docs/api-reference/images/create#images/create-n
   */
  n?: number;

  /**
   * The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
   * https://platform.openai.com/docs/api-reference/images/create#images/create-size
   */
  size?: "256x256" | "512x512" | "1024x1024";

  /**
   * The format in which the generated images are returned. Must be one of url or b64_json.
   * https://platform.openai.com/docs/api-reference/images/create#images/create-response_format
   */
  responseFormat?: "url" | "b64_json";

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/images/create#images/create-user
   */
  user?: string;
}

export interface ImageEditOptions {
  /**
   * The image to edit. Must be a valid PNG file, less than 4MB, and square.
   * If mask is not provided, image must have transparency, which will be used as the mask.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-image
   */
  image: FileSpecifier;

  /**
   * An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited.
   * Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-mask
   */
  mask?: string;

  /**
   * A text description of the desired image(s). The maximum length is 1000 characters.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-prompt
   */
  prompt: string;

  /**
   * The number of images to generate. Must be between 1 and 10.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-n
   */
  n?: number;

  /**
   * The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-size
   */
  size?: "256x256" | "512x512" | "1024x1024";

  /**
   * The format in which the generated images are returned. Must be one of url or b64_json.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-response_format
   */
  responseFormat?: "url" | "b64_json";

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/images/create-edit#images/create-edit-user
   */
  user?: string;
}

export interface ImageVariationOptions {
  /**
   * The image to edit. Must be a valid PNG file, less than 4MB, and square.
   * https://platform.openai.com/docs/api-reference/images/create-variation#images/create-variation-image
   */
  image: FileSpecifier;

  /**
   * The number of images to generate. Must be between 1 and 10.
   * https://platform.openai.com/docs/api-reference/images/create-variation#images/create-variation-n
   */
  n?: number;

  /**
   * The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
   * https://platform.openai.com/docs/api-reference/images/create-variation#images/create-variation-size
   */
  size?: "256x256" | "512x512" | "1024x1024";

  /**
   * The format in which the generated images are returned. Must be one of url or b64_json.
   * https://platform.openai.com/docs/api-reference/images/create-variation#images/create-variation-response_format
   */
  responseFormat?: "url" | "b64_json";

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/images/create-variation#images/create-variation-user
   */
  user?: string;
}

export interface EmbeddingsOptions {
  /**
   * ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
   * https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-model
   */
  model: string;

  /**
   * Input text to get embeddings for, encoded as a string or array of tokens.
   * To get embeddings for multiple inputs in a single request, pass an array of strings or array of token arrays.
   * Each input must not exceed 8192 tokens in length.
   * https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-input
   */
  input: string | string[];

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
   * https://platform.openai.com/docs/api-reference/embeddings/create#embeddings/create-user
   */
  user?: string;
}

export interface TranscriptionOptions {
  /**
   * The audio file to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-file
   */
  file: FileSpecifier;

  /**
   * ID of the model to use. Only whisper-1 is currently available.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-model
   */
  model: string;

  /**
   * An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-prompt
   */
  prompt?: string;

  /**
   * The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-response_format
   */
  responseFormat?: "json" | "text" | "srt" | "verbose_json" | "vtt";

  /**
   * The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
   * If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-temperature
   */
  temperature?: number;

  /**
   * The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-language
   */
  language?: string;
}

export interface TranslationOptions {
  /**
   * The audio file to translate, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-file
   */
  file: FileSpecifier;

  /**
   * ID of the model to use. Only whisper-1 is currently available.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-model
   */
  model: string;

  /**
   * An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-prompt
   */
  prompt?: string;

  /**
   * The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-response_format
   */
  responseFormat?: "json" | "text" | "srt" | "verbose_json" | "vtt";

  /**
   * The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
   * If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-temperature
   */
  temperature?: number;

  /**
   * The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
   * https://platform.openai.com/docs/api-reference/audio/create#audio/create-language
   */
  language?: string;
}

export interface FineTuneOptions {
  /**
   * The ID of an uploaded file that contains training data.
   * Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys "prompt" and "completion".
   * Additionally, you must upload your file with the purpose fine-tune.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-training_file
   */
  trainingFile: string;

  /**
   * The ID of an uploaded file that contains validation data.
   * If you provide this file, the data is used to generate validation metrics periodically during fine-tuning.
   * These metrics can be viewed in the fine-tuning results file. Your train and validation data should be mutually exclusive.
   * Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys "prompt" and "completion".
   * Additionally, you must upload your file with the purpose fine-tune.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-training_file
   */
  validationFile: string;

  /**
   * The name of the base model to fine-tune.
   * You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21.
   * To learn more about these models, see the Models documentation.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-model
   */
  model: string;

  /**
   * The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-n_epochs
   */
  nEpochs?: number;

  /**
   * The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.
   * By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general,
   * we've found that larger batch sizes tend to work better for larger datasets.
   */
  batchSize?: number;

  /**
   * The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.
   * By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final batch_size (larger learning rates tend to perform better with larger batch sizes).
   * We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-learning_rate_multiplier
   */
  learningRateMultiplier?: number;

  /**
   * The weight to use for loss on the prompt tokens.
   * This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0),
   * and can add a stabilizing effect to training when completions are short.
   * If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-prompt_loss_weight
   */
  promptLossWeight?: number;

  /**
   * If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch.
   * These metrics can be viewed in the results file.
   * In order to compute classification metrics, you must provide a validation_file.
   * Additionally, you must specify classification_n_classes for multiclass classification or classification_positive_class for binary classification.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-compute_classification_metrics
   */
  computeClassificationMetrics?: boolean;

  /**
   * The number of classes in a classification task.
   * This parameter is required for multiclass classification.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_n_classes
   */
  classificationNClasses?: number;

  /**
   * The positive class in binary classification.
   * This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_positive_class
   */
  classificationPositiveClass?: string;

  /**
   * If this is provided, we calculate F-beta scores at the specified beta values.
   * The F-beta score is a generalization of F-1 score. This is only used for binary classification.
   * With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight.
   * A larger beta score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision and less on recall.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-classification_betas
   */
  classificationBetas?: number[];

  /**
   * A string of up to 40 characters that will be added to your fine-tuned model name.
   * For example, a suffix of "custom-model-name" would produce a model name like ada:ft-your-org:custom-model-name-2022-02-15-04-21-04.
   * https://platform.openai.com/docs/api-reference/fine-tunes/create#fine-tunes/create-suffix
   */
  suffix?: string;
}

export interface Model {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  permission: {
    id: string;
    object: "model_permission";
    created: number;
    allow_create_engine: boolean;
    allow_sampling: boolean;
    allow_logprobs: boolean;
    allow_search_indices: boolean;
    allow_view: boolean;
    allow_fine_tuning: boolean;
    organization: string;
    group: null | string;
    is_blocking: boolean;
  }[];
  root: string;
  parent: null | string;
}

export interface ModelList {
  object: "list";
  data: Model[];
}

export interface Completion {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: {
    text: string;
    index: number;
    logprobs: number | null;
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface CompletionStream {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: {
    text: string;
    index: number;
    logprobs: number | null;
    finish_reason: string;
  }[];
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  choices: {
    index: number;
    message: ChatCompletionMessage;
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ChatCompletionStreamDelta {
  name?: string;
  role?: "system" | "assistant" | "user";
  content?: string | null;
  function_call?: {
    name?: string;
    arguments: string;
  }
}

export interface ChatCompletionStream {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  choices: {
    index: number;
    delta: ChatCompletionStreamDelta;
    finish_reason: string | null;
  }[];
}

export interface Edit {
  object: "edit";
  created: number;
  choices: {
    text: string;
    index: number;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface Image {
  created: number;
  data: {
    url: string;
    b64_json: string;
  }[];
}

export interface Embedding {
  object: "list";
  data: {
    object: "embedding";
    embedding: number[];
    index: number;
  }[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface Transcription {
  text: string;
}

export interface Translation {
  text: string;
}

export interface FileInstance {
  id: string;
  object: "file";
  bytes: number;
  created_at: number;
  filename: string;
  purpose: string;
}

export interface FileList {
  data: FileInstance[];
  object: "list";
}

export interface DeletedFile {
  id: string;
  object: "file";
  deleted: boolean;
}

export interface FineTuneEvent {
  object: "fine-tune-event";
  created_at: number;
  level: string;
  message: string;
}

export interface FineTuneEventList {
  object: "list";
  data: FineTuneEvent[];
}

export interface FineTune {
  id: string;
  object: "fine-tune";
  model: string;
  created_at: number;
  fine_tuned_model: null | string;
  hyperparams: {
    batch_size: number;
    learning_rate_multiplier: number;
    n_epochs: number;
    prompt_loss_weight: number;
  };
  organization_id: string;
  result_files: FileInstance[];
  status: "pending" | "succeeded" | "cancelled";
  validation_files: FileInstance[];
  training_files: FileInstance[];
  updated_at: number;
}

export interface FineTuneList {
  object: "list";
  data: FineTune[];
}

export interface DeletedFineTune {
  id: string;
  object: "model";
  deleted: boolean;
}

export interface Moderation {
  id: string;
  model: string;
  results: {
    categories: {
      hate: boolean;
      "hate/threatening": boolean;
      "self-harm": boolean;
      sexual: boolean;
      "sexual/minors": boolean;
      violence: boolean;
      "violence/graphic": boolean;
    };
    category_scores: {
      hate: number;
      "hate/threatening": number;
      "self-harm": number;
      sexual: number;
      "sexual/minors": number;
      violence: number;
      "violence/graphic": number;
    };
    flagged: boolean;
  }[];
}
