rdagent/scenarios/qlib/experiment/factor_data_template/generate.py
生成数据

rdagent/app/qlib_rd_loop/quant.py  
入口
QuantRDLoop  初始化及每个步骤

rdagent/utils/workflow/loop.py
主管循环
假设 - 转实验描述 - 开发 - 反馈

rdagent/components/coder/CoSTEER/knowledge_management.py
CoSTEERKnowledgeBaseV2  知识库





direct_exp_gen
    rdagent/components/proposal/__init__.py LLMHypothesisGen  生成假设
    rdagent/components/proposal/__init__.py Hypothesis2Experiment - LLMHypothesis2Experiment — FactorHypothesis2Experiment - QlibFactorHypothesis2Experiment 假设转实验描述


coding
    rdagent/components/coder/factor_coder/__init__.py Developer - CoSTEER - FactorCoSTEER   代码开发
        rdagent/components/coder/CoSTEER/__init__.py  CoSTEER
            rdagent/core/evolving_agent.py EvoAgent -RAGEvoAgent 进化 agent
                1、查RAG
                2、rdagent/components/coder/CoSTEER/evolving_strategy.py  MultiProcessEvolvingStrategy 分任务写代码
                    rdagent/components/coder/factor_coder/evolving_strategy.py  FactorMultiProcessEvolvingStrategy 写代码
                3、rdagent/components/coder/CoSTEER/evaluators.py CoSTEERMultiEvaluator 分评审任务
                    rdagent/components/coder/factor_coder/evaluators.py FactorEvaluatorForCoder 因子代码评审
                        rdagent/components/coder/factor_coder/factor.py FactorFBWorkspace 执行代码
                        rdagent/components/coder/factor_coder/eva_utils.py FactorCodeEvaluator  代码评估
                        rdagent/components/coder/factor_coder/eva_utils.py FactorFinalDecisionEvaluator 最终评价

running
    rdagent/scenarios/qlib/developer/factor_runner.py QlibFactorRunner
        rdagent/scenarios/qlib/developer/utils.py  process_factor_data
            rdagent/components/coder/factor_coder/factor.py FactorFBWorkspace  执行代码，参数不同，会跑所有数据
            rdagent/scenarios/qlib/experiment/workspace.py  QlibFBWorkspace 执行模型测试  conf_baseline.yaml  conf_combined_factors.yaml
            
feedback



RD2bench.json
example.json



rdagent/scenarios/qlib/developer/utils.py  todo 修改all