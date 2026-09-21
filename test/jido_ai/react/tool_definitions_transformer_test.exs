defmodule Jido.AI.Reasoning.ReAct.ToolDefinitionsTransformerTest do
  use ExUnit.Case, async: false
  use Mimic

  alias Jido.AI.Reasoning.ReAct.{Config, Runner, State}

  defmodule Probe do
    use Jido.Action,
      name: "probe",
      description: "Original description",
      schema: Zoi.object(%{value: Zoi.integer()})

    def run(%{value: value}, _context), do: {:ok, %{value: value}}
  end

  defmodule LegacyTransformer do
    @behaviour Jido.AI.Reasoning.ReAct.RequestTransformer
    @impl true
    def transform_request(request, _state, _config, _context) do
      tools = Enum.map(request.llm_opts[:tools], &%{&1 | description: "Overwritten during regeneration"})
      {:ok, %{llm_opts: [tools: tools]}}
    end
  end

  defmodule Transformer do
    @behaviour Jido.AI.Reasoning.ReAct.RequestTransformer
    @impl true
    defdelegate transform_request(request, state, config, context), to: LegacyTransformer

    @impl true
    def transform_tool_definitions(tools, state, config, context) do
      send(context.owner, {:callback, tools, state, config})
      context.transform.(tools)
    end
  end

  setup :set_mimic_from_context

  for streaming <- [true, false] do
    @streaming streaming

    test "final definitions reach the provider, streaming=#{streaming}" do
      parent = self()
      provider = if @streaming, do: :stream_text, else: :generate_text

      Mimic.expect(ReqLLM.Generation, provider, fn _model, _messages, opts ->
        send(parent, {:provider_tools, opts[:tools]})
        {:error, :offline_provider_boundary}
      end)

      config = config(@streaming, Transformer)
      transform = fn tools -> {:ok, Enum.map(tools, &%{&1 | description: "Request description"})} end
      events = run(config, transform)

      assert_received {:callback, original, %State{}, ^config}
      assert [%ReqLLM.Tool{description: "Original description"}] = original
      assert_received {:provider_tools, [tool]}
      assert tool.description == "Request description"
      assert Map.delete(tool, :description) == Map.delete(hd(original), :description)
      assert config.tools == %{"probe" => Probe}
      assert Enum.any?(events, &(&1.kind == :request_failed))
    end

    for transformer <- [nil, LegacyTransformer] do
      @transformer transformer
      test "generated definitions are unchanged without the optional callback, streaming=#{streaming}, transformer=#{inspect(transformer)}" do
        parent = self()
        provider = if @streaming, do: :stream_text, else: :generate_text
        config = config(@streaming, @transformer)
        expected = Config.reqllm_tools(config)

        Mimic.expect(ReqLLM.Generation, provider, fn _model, _messages, opts ->
          send(parent, {:provider_tools, opts[:tools]})
          {:error, :offline_provider_boundary}
        end)

        run(config, nil)
        assert_received {:provider_tools, ^expected}
        refute_received {:callback, _, _, _}
      end
    end

    test "callback errors stop before contacting the provider, streaming=#{streaming}" do
      reject_provider_calls()
      events = run(config(@streaming, Transformer), fn _ -> {:error, :unavailable} end)
      assert_failure(events, {:tool_definitions_transformer, :unavailable})
    end

    test "invalid callback results stop before contacting the provider, streaming=#{streaming}" do
      reject_provider_calls()

      for transform <- [
            fn _ -> :invalid end,
            fn _ -> {:ok, nil} end,
            fn _ -> {:ok, [%{}]} end,
            fn _ -> {:ok, []} end,
            fn [tool] -> {:ok, [%{tool | name: "renamed"}]} end,
            fn [tool] -> {:ok, [tool, tool]} end
          ] do
        events = run(config(@streaming, Transformer), transform)
        failure = Enum.find(events, &(&1.kind == :request_failed))
        assert failure.data.error_type == :request_transform
        assert {:invalid_tool_definitions_transformer_result, _} = failure.data.error
      end
    end

    test "callback exceptions become request errors, streaming=#{streaming}" do
      reject_provider_calls()
      events = run(config(@streaming, Transformer), fn _ -> raise "description lookup failed" end)

      assert_failure(
        events,
        {:tool_definitions_transformer_exception, %{error: "description lookup failed", type: RuntimeError}}
      )
    end
  end

  test "definitions are transformed on each turn while actions keep their execution mapping" do
    config = config(false, Transformer)

    Mimic.expect(ReqLLM.Generation, :generate_text, fn _, _, opts ->
      assert [%ReqLLM.Tool{description: "Request description"}] = opts[:tools]

      {:ok,
       %{
         message: %{content: "", tool_calls: [%{id: "call_probe", name: "probe", arguments: %{value: 42}}]},
         finish_reason: :tool_calls
       }}
    end)

    Mimic.expect(ReqLLM.Generation, :generate_text, fn _, _, opts ->
      assert [%ReqLLM.Tool{description: "Request description"}] = opts[:tools]
      {:ok, %{message: %{content: "Finished", tool_calls: []}, finish_reason: :stop}}
    end)

    events =
      run(config, fn tools ->
        {:ok,
         Enum.map(tools, &%{&1 | description: "Request description", callback: fn _ -> {:error, :wrong_registry} end})}
      end)

    assert_received {:callback, _, %State{}, ^config}
    assert_received {:callback, _, %State{}, ^config}
    refute_received {:callback, _, _, _}
    completed = Enum.find(events, &(&1.kind == :tool_completed))
    assert completed.data.result == {:ok, %{value: 42}, []}
    assert Enum.any?(events, &(&1.kind == :request_completed and &1.data.result == "Finished"))
  end

  test "an empty tool selection can pass through the callback" do
    config = Config.new(model: :capable, tools: [], streaming: false, request_transformer: Transformer)

    Mimic.expect(ReqLLM.Generation, :generate_text, fn _, _, opts ->
      assert opts[:tools] == []
      {:error, :offline_provider_boundary}
    end)

    run(config, fn [] -> {:ok, []} end)
    assert_received {:callback, [], %State{}, ^config}
  end

  defp config(streaming, transformer) do
    Config.new(model: :capable, tools: [Probe], streaming: streaming, request_transformer: transformer)
  end

  defp run(config, transform) do
    Runner.stream("Hello", config, context: %{owner: self(), transform: transform}) |> Enum.to_list()
  end

  defp reject_provider_calls do
    for provider <- [:stream_text, :generate_text] do
      Mimic.reject(ReqLLM.Generation, provider, 3)
    end
  end

  defp assert_failure(events, reason) do
    failure = Enum.find(events, &(&1.kind == :request_failed))
    assert failure.data.error_type == :request_transform
    assert failure.data.error == reason
  end
end
