defmodule Jido.AI.Reasoning.ReAct.RequestTransformer do
  @moduledoc """
  Behavior for advanced per-turn ReAct request shaping.

  A request transformer can inspect the current runtime state and tool context
  before each LLM turn, then return request overrides.

  This is intended for patterns such as:

  - request-scoped tool gating
  - dynamic structured-output schemas
  - provider-specific `llm_opts` based on tool results
  - custom message projection beyond the default context rendering
  - **per-turn model selection** — returning `model:` in the overrides swaps
    which provider handles the next LLM turn. `llm_opts.provider_options`
    overrides are re-validated against the selected model's provider schema,
    so xAI-only keys (e.g. `xai_api`) can be added on xAI turns without
    breaking Fireworks turns on the same agent.

  For normal ReAct turns, the runtime regenerates `llm_opts[:tools]` from the
  returned `tools` field so the exposed LLM tools and execution registry stay
  aligned. To change the generated definitions, implement the optional
  `transform_tool_definitions/4` callback. It runs after tool selection and
  regeneration, before ReqLLM encodes the provider request, for both streaming
  and non-streaming turns.

  Structured-output repair turns also pass through `transform_request/4`. Their
  request contains the repair prompt and an empty tool set. The runtime applies
  `messages`, `model`, and `llm_opts` overrides, but it keeps tools disabled for
  the repair call and does not call `transform_tool_definitions/4`.
  """

  alias Jido.AI.Reasoning.ReAct.{Config, State, ToolSelection}

  @type request :: %{
          required(:messages) => [map()],
          required(:llm_opts) => keyword(),
          required(:tools) => ToolSelection.tools_input(),
          required(:model) => term()
        }

  @type overrides :: %{
          optional(:messages) => [map()],
          optional(:llm_opts) => keyword() | map(),
          optional(:tools) => ToolSelection.tools_input(),
          optional(:model) => term()
        }

  @doc """
  Validate a request transformer module.
  """
  @spec validate(module() | nil) ::
          {:ok, module() | nil}
          | {:error, :invalid_request_transformer}
          | {:error, {:request_transformer_not_loaded, module()}}
          | {:error, {:request_transformer_missing_callback, module()}}
  def validate(nil), do: {:ok, nil}

  def validate(module) when is_atom(module) do
    cond do
      not Code.ensure_loaded?(module) ->
        {:error, {:request_transformer_not_loaded, module}}

      not function_exported?(module, :transform_request, 4) ->
        {:error, {:request_transformer_missing_callback, module}}

      true ->
        {:ok, module}
    end
  end

  def validate(_other), do: {:error, :invalid_request_transformer}

  @doc """
  Fingerprint a validated transformer for checkpoint compatibility.
  """
  @spec fingerprint(module() | nil) :: String.t()
  def fingerprint(nil), do: ""
  def fingerprint(module) when is_atom(module), do: Atom.to_string(module)

  @callback transform_request(request(), State.t(), Config.t(), map()) ::
              {:ok, overrides()} | {:error, term()}

  @doc """
  Transform the final tool definitions sent to the LLM on each turn.

  Receives the regenerated `ReqLLM.Tool` structs, current runtime state, original
  config, and the same runtime context passed to `transform_request/4`. This can
  customize descriptions or parameter schemas without replacing the action
  execution registry. Use `transform_request/4` to change which actions are available.

  Return `{:ok, tools}` with one `ReqLLM.Tool` for each input tool, preserving
  names. Order may change. Adding, removing, renaming, or duplicating tools is
  rejected. The callback is responsible for valid descriptions and schemas.
  Modules that omit this callback retain the generated definitions unchanged.

  `{:error, reason}`, malformed results, and exceptions terminate the request
  with a `:request_transform` error before contacting the provider.
  """
  @callback transform_tool_definitions([ReqLLM.Tool.t()], State.t(), Config.t(), map()) ::
              {:ok, [ReqLLM.Tool.t()]} | {:error, term()}

  @optional_callbacks transform_tool_definitions: 4
end
