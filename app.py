from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__, static_url_path='/static/')

# Business logic
validation_days = 2
validations_per_day = 1 / validation_days
validations_per_month = int(30 * validations_per_day)
validations_per_year = int(365 * validations_per_day)


# (paste the calc_rewards and calcCAGR functions here)

def calc_rewards(market_size=10000, validator_reward_rate=0.02, starting_power=1000, compounding=False):
    pct_market_participation = 0.5
    reward_bonus_tiers = np.array([0.15, 0.25, 0.30])

    rewards_df = pd.DataFrame(index=np.arange(0, validations_per_year + 1))  
    rewards_df['marketSize'] = np.random.randint(market_size, market_size+1, size=validations_per_year + 1)
    rewards_df['powerNeeded'] = rewards_df['marketSize'] * pct_market_participation

    rewards_df.loc[0, 'endingPower'] = starting_power

    for validation in range(1, validations_per_year + 1):
        if compounding:
            starting_power = rewards_df.loc[validation - 1, 'endingPower']

        rewards_df.loc[validation, 'pctOfTotalPower'] = starting_power / rewards_df.loc[validation, 'powerNeeded']
        rewards_df.loc[validation, 'rewardFromValidation'] = (market_size * validator_reward_rate) * rewards_df.loc[validation, 'pctOfTotalPower']
        rewards_df.loc[validation, 'rewardBonusFromTier'] = reward_bonus_tiers[min(validation // validations_per_month, len(reward_bonus_tiers) - 1)]
        rewards_df.loc[validation, 'rewardBonusFromTierAmt'] = rewards_df.loc[validation, 'rewardBonusFromTier'] * rewards_df.loc[validation, 'rewardFromValidation']
        rewards_df.loc[validation, 'rewardTotal'] = rewards_df.loc[validation, 'rewardFromValidation'] + rewards_df.loc[validation, 'rewardBonusFromTierAmt']

        rewards_df.loc[validation, 'endingPower'] = starting_power + rewards_df.loc[validation, 'rewardTotal']

    rewards_df['rewardTotalCumSum'] = rewards_df['rewardTotal'].cumsum()
    rewards_df['rewardTotalCumSum'] = rewards_df['rewardTotalCumSum'].round(2)  # Round to 2 decimal places

    return rewards_df

def calcCAGR(df):
    first_value = df.loc[0, 'endingPower']
    last_value = df.loc[df.shape[0] - 1, 'endingPower']
    
    # Calculate CAGR
    cagr = ((last_value / first_value) ** (1 / (df.shape[0] - 1))) - 1
    print(f'Compounded Annual Growth: {cagr * 100}%')
    
    return cagr * 100

def convert_to_percent(cagr):
    return str(cagr) + '%'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        market_size = float(request.form.get('market_size'))
        # validator_reward_rate = float(request.form.get('validator_reward_rate'))
        starting_power = float(request.form.get('starting_power'))

        # # Compute the rewards
        # compounding_rewards_df = calc_rewards(market_size, validator_reward_rate, starting_power, compounding=True)
        # non_compounding_rewards_df = calc_rewards(market_size, validator_reward_rate, starting_power, compounding=False)
        
        # Compute the rewards
        compounding_rewards_df = calc_rewards(market_size=market_size, starting_power=starting_power, compounding=True)
        non_compounding_rewards_df = calc_rewards(market_size=market_size, starting_power=starting_power, compounding=False)

        # Calculate the CAGR
        compounding_cagr = calcCAGR(compounding_rewards_df)
        non_compounding_cagr = calcCAGR(non_compounding_rewards_df)

        # Prepare the results matrix
        results_matrix = pd.DataFrame(index=['Without Compounding', 'With Compounding'], columns=['After 1 Validation', 'After 1 Month', 'After 1 Year', 'CAGR'])
        results_matrix.loc['Without Compounding', 'After 1 Validation'] = round(non_compounding_rewards_df.loc[1, 'rewardTotalCumSum'], 2)
        results_matrix.loc['Without Compounding', 'After 1 Month'] = round(non_compounding_rewards_df.loc[validations_per_month, 'rewardTotalCumSum'], 2)
        results_matrix.loc['Without Compounding', 'After 1 Year'] = round(non_compounding_rewards_df.loc[validations_per_year, 'rewardTotalCumSum'], 2)
        results_matrix.loc['Without Compounding', 'CAGR'] = round(non_compounding_cagr, 2)
        results_matrix.loc['With Compounding', 'After 1 Validation'] = round(compounding_rewards_df.loc[1, 'rewardTotalCumSum'], 2)
        results_matrix.loc['With Compounding', 'After 1 Month'] = round(compounding_rewards_df.loc[validations_per_month, 'rewardTotalCumSum'], 2)
        results_matrix.loc['With Compounding', 'After 1 Year'] = round(compounding_rewards_df.loc[validations_per_year, 'rewardTotalCumSum'], 2)
        results_matrix.loc['With Compounding', 'CAGR'] = round(compounding_cagr, 2)

        # Format numbers with commas for thousands
        results_matrix = results_matrix.applymap('{:,.2f}'.format)
        results_matrix['CAGR'] = results_matrix['CAGR'].apply(convert_to_percent)
        print(results_matrix)

        headings1 = tuple(results_matrix.columns)
        headings2 = tuple(compounding_rewards_df.columns)
        headings3 = tuple(non_compounding_rewards_df.columns)

        print(results_matrix.index)
        result_matrix = list(results_matrix.itertuples(index=True))
        comp_rew_df = list(compounding_rewards_df.itertuples(index=True))
        non_comp_rew_df = list(non_compounding_rewards_df.itertuples(index=True))
        return render_template('results.html', index_name=' ', headings=[headings1,headings2,headings3], data=[result_matrix,comp_rew_df,non_comp_rew_df])
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
