import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal
import time

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        2  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Step 1: Generate 5 key factors
            key_factors_prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster. 
                The superforecaster will give you a question they intend to forecast on.  
                When possible, try to get a diverse range of perspectives if the question is controversial. 
                Use your judgment in deciding the most relevant information. 
                You do not produce forecasts yourself - you are responsible for coming up with Google search queries to find relevant information.

                Tips for identifying key factors:
                - Consider historical trends and base rates.
                - Identify major drivers or variables that could influence the outcome.
                - Think about external events, policies, or technological changes.
                - Be creative and think outside the box.
                - Consider the implications of the question and its context.
                - Be sure to include the status quo outcome and the current trend. (Factors 1 and 2 at least)

                The question is:
                {question.question_text}

                Your task is to list 5 key factors that could influence the outcome of this question.
                Provide the factors as a numbered list with a brief explanation for each.

                The last thing you write is:
                Factor_1 :=> google search query for Factor_1
                Factor_2 :=> google search query for Factor_2
                ...
                Factor_5 :=> google search query for Factor_5
                """
            )
            key_factors_response = await GeneralLlm(
                model="metaculus/o4-mini", temperature=1
            ).invoke(key_factors_prompt)
            logger.info(
                f"Key factors for question {question.page_url}:\n{key_factors_response}"
            )

            # Extract the 5 factors from the response (assuming they are in a numbered list)
            key_factors = key_factors_response.split("\n")[:5]
            for i in range(len(key_factors)):
                # Remove the numbering and any leading/trailing whitespace
                key_factors[i] = key_factors[i].split(":=>")[-1].strip()

            # Step 2: Research each factor using AskNews
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                general_research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )

                research = ""
                for factor in key_factors:
                    try:
                        factor_research = (
                            await AskNewsSearcher().get_formatted_news_async(factor)
                        )
                        research += f"Research on '{factor}':\n{factor_research}\n\n"
                    except Exception as e:
                        logger.warning(f"Error researching factor '{factor}': {e}")

                    time.sleep(1)  # Rate limit for AskNews API
            else:
                logger.warning("AskNews credentials not found. Skipping research.")

            full_research = f"""Research for question {question.page_url}:
                        Here is the general background information you will require to answer the question. The majority of your reasoning will be based around this information:
                        {general_research}

                        Your assistant has also found some relevant information on factors that could influence the outcome of this question. This information is not required to answer the question, but should definitely be considered as it will set you apart:
                        {research}"""
            logger.info(f"Research for question {question.page_url}:\n{full_research}")
            return full_research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Break down the question into its components and think about the implications of each component. Ask yourself: what would need to happen for this question to be true?

            If we are thinking about an event 100 years into the future you should step into it year by year. Base yourself on fact as much as possible.
            If you are thinking about how 100 people might behave, you should think first about how 99 of them would behave, then 98 of them, and so on.

            Be conservative in your estimates. If you are not sure, it is better to be conservative and give a lower estimate than to be overly optimistic.

            While you are in the interview the single most important think you can do is reason out loud and explain your reasoning to the interviewer. This allows you to show your reasoning and helps the interviewer understand your thought process. If you make a mistake this allows you to correct it and show that you are a good forecaster.

            Your interview question is:
            {question.question_text}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research gathered by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.


            Here are top tips from good forecasters:

            (1) If an event was anticipated to happen in a certain timeframe, but 80% of that time has passed and there is no recent news or updates about it happening soon, then you should be skeptical that it will happen on the originally stated timeframe. It probably means that it will be delayed or plans have changed. Some examples:
            If Elon Musk says definitely that robotaxis will be available in 1 year, but after 11 months there have not been any specific public updates confirming that robotaxis will be available then it is highly unlikely to happen in the remaining month.
            If Donald Trump says that he will definitely have a deal signed in 90 days, but 70 days have passed without updates on the progress, then it is highly unlikely there will be a deal in the remaining 20 days.
            
            (2) Think about base rates for similar events in the past. Sometimes a base rate is the best you can do, if you can’t find much information about the question. Example questions where the base rate is a particularly good starting point include:
            Will the temperature in Miami exceed 100 degrees next month?
            Will there be an earthquake of magnitude 5 or more in San Francisco in 2026?
            
            (3) Sometimes finding a good base rate is difficult, especially when the events are relatively unique. In these cases, you need to put more weight on the “inside view” which means weighing considerations that appear specific to the situation that lead to a very different forecast than base rates would dictate. You will need to use your own judgment.
            
            (4) Put extra weight on the status quo outcome since the world changes slowly most of the time. This is especially true when coordination or agreement between people or organizations is required. For example, signing multi-national treaties and passing legislation often take longer than one might imagine from reading the news.
            
            (5) Think about if there are seasonal effects. For example, the sales of homes or travel are likely to have seasonal patterns.
            
            (6) Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter. Other trends have momentum, like the number of COVID cases from day to day.
            
            (7) Think about the scope of the question.
            
            (8) Think about the incentives and power of any influential people involved in the situation. For example, Putin has the power to single-handedly dictate Russian military or diplomatic response.

            (9) Sometimes there are multiple data sources for the same number that have very different values. For example, Trading Economics reports 1.84B UAH for Ukrainian debt while the IMF reports a value of 7B UAH. These are both reputable sources, but they are using different definitions in their reporting. It is important to focus on the source used in the resolution criteria.
            
            (10) Pre-mortem. Think about how you are most likely to be wrong. Imagine that you are writing a letter to your future self that you will open and read once the outcome is known. In the letter you try to explain to your future self the most likely way that your forecast will be deemed to be a poor forecast. Are you most worried about missing a key piece of information? What is the biggest uncertainty and would keep you awake at night?

            (11) More general advice:
            - Even if something seems impossible, never forecast less than 3%. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Even if something seems certain, never forecast more than 97%. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - There are exceptions to all of the above tips. And, sometimes the tips lead you in conflicting directions, so don't take them too strictly.
            - Like a good forecaster, you should use your own judgment to come to the most accurate forecast! 

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) Think about answering the question with different scopes to help ensure that you have a self consistent view and have considered the broader context. 
                For example, imagine the question is: Will a company declare bankruptcy in the next 3 months? It can be useful to force yourself to forecast the probability of bankruptcy over the next 1 year, 2 years, and 5 years. Doing this in a self-consistent way helps to force you to consider the scope explicitly.
                If you forecast 40% in 12 months, then you might forecast 10% in 3 months to be scope sensitive. However, it is also possible that the bankruptcy risk is higher in the near term, so it could still be 25%.
                You will need to use your judgment. Being scope insensitive is a common cognitive bias and this exercise is meant to help combat this bias by forcing you to explicitly consider the question’s scope.
            (d) A brief description of a scenario that results in a No outcome.
            (e) A brief description of a scenario that results in a Yes outcome.
            
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Break down the question into its components and think about the implications of each component. Ask yourself: what would need to happen for this question to be true?

            If we are thinking about an event 100 years into the future you should step into it year by year. Base yourself on fact as much as possible.
            If you are thinking about how 100 people might behave, you should think first about how 99 of them would behave, then 98 of them, and so on.

            Be conservative in your estimates. If you are not sure, it is better to be conservative and give a lower estimate than to be overly optimistic.

            While you are in the interview the single most important think you can do is reason out loud and explain your reasoning to the interviewer. This allows you to show your reasoning and helps the interviewer understand your thought process. If you make a mistake this allows you to correct it and show that you are a good forecaster.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research provided by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            Tips from good forecasters:
            - Think about the base rates for similar events in the past.
            - Put extra weight on the status quo outcome since the world changes slowly most of the time. 
            - Think about if there are seasonal effects.
            - Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter.
            - Even if an option seems impossible, never put less than 2% on an option. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Even if an option seems certain, never put more than 95% on an option. (It is possible that you don't have all of the information, or have misunderstood something.)            
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - Like a good forecaster, you should use your own judgment to come to the most accurate forecast.

            The last thing you write is your final probabilities for the options in this order {question.options} as:

            FINAL FORECAST
            [text of first option]: [Probability of first option] %
            [text of second option]: [Probability of second option] %
            ...
            [text of last option]: [Probability of last option] %

            For [text of first option], replace it with the actual text from this list: {question.options}.
            For [Probability first option], replace it with the probability of the first option. It should be an integer 1-99.
            Do not write any text after the percent sign for your probability of an option.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Break down the question into its components and think about the implications of each component. Ask yourself: what would need to happen for this question to be true?

            If we are thinking about an event 100 years into the future you should step into it year by year. Base yourself on fact as much as possible.
            If you are thinking about how 100 people might behave, you should think first about how 99 of them would behave, then 98 of them, and so on.

            Be conservative in your estimates. If you are not sure, it is better to be conservative and give a lower estimate than to be overly optimistic.

            While you are in the interview the single most important think you can do is reason out loud and explain your reasoning to the interviewer. This allows you to show your reasoning and helps the interviewer understand your thought process. If you make a mistake this allows you to correct it and show that you are a good forecaster.

            Your interview question is:
            {question.question_text}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research provided by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) A brief description of an unexpected scenario that results in a low outcome.
            (e) A brief description of an unexpected scenario that results in a high outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters err on the side of having wide confidence intervals to account for unexpected outcomes.

            Important tips: 
            - Your 20th percentile forecast should be greater than the lower bound of {question.lower_bound}.
            - Your 80th percentile forecast should be less than the upper bound of {question.upper_bound}.
            - Have a wide range for your tails since you might not have all the information, or might be misunderstanding something.
            - Historically your 80/20 confidence interval has been much too narrow. To be more calibrated make your P10 to P90 4x wider than your P20 to P80 interval. These wide tails will help your calibration.
            - Think about if there are seasonal effects.
            - Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter.
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - Like a good forecaster, use your own judgment!

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.

            The last thing you write is your final answer as:
            
            FINAL FORECAST
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(model="metaculus/o4-mini", temperature=1)
        },
    )

    if run_mode == "tournament":
        template_bot.skip_previously_forecasted_questions = True
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            # "https://www.metaculus.com/questions/31273/will-a-new-war-kill-at-least-5000-people-in-2025/",
            # "https://www.metaculus.com/questions/22954/crypto-to-surpass-gold-before-2035/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # "https://www.metaculus.com/questions/21017/next-pope/",
            # "https://www.metaculus.com/questions/37321/who-will-be-the-2028-republican-nominee-for-president-of-the-united-states/"
            # "https://www.metaculus.com/questions/36460/number-of-times-starship-reaches-160km-in-2025/"
            # "https://www.metaculus.com/questions/37151/george-santos-sentence-length/",
            # "https://www.metaculus.com/questions/34751/home-battery-annual-frequency-imbalance-return/",
            # "https://www.metaculus.com/questions/36881/medicaid-cut-over-10-years-in-2025-reconciliation-bill/",
            # "https://www.metaculus.com/questions/26327/us-measles-outbreak-2025/",
            # "https://www.metaculus.com/questions/605/global-warming-in-2100-over-1880-baseline/",
            # "https://www.metaculus.com/questions/9062/time-from-weak-agi-to-superintelligence/",
            # "https://www.metaculus.com/questions/31817/h5-case-fatality-rate-in-us/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
