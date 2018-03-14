"""
This tool uses the data provided by the Kaggle Machine Learning Mania challenge
and generates predictions for March Madness brackets.

More about the competition:
https://www.kaggle.com/c/march-machine-learning-mania-2017
"""
import sys, getopt, datetime, math, csv, random, os
import pandas as pd
import numpy
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

base_elo = 1600
team_elos = {}  # Reset each year.
team_stats = {}
X = []
y = []
submission_data = []

stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr',
               'ast', 'to', 'stl', 'blk', 'pf']


def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    """
    This is originally from from:
    http://zurb.com/forrst/posts/An_Elo_Rating_function_in_Python_written_for_foo-hQl
    """
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def initialize_data(prediction_year):
    for i in range(1985, prediction_year+1):
        team_elos[i] = {}
        team_stats[i] = {}


def get_elo(season, team):
    try:
        return team_elos[season][team]
    except:
        try:
            # Get the previous season's ending value.
            team_elos[season][team] = team_elos[season-1][team]
            return team_elos[season][team]
        except:
            # Get the starter elo.
            team_elos[season][team] = base_elo
            return team_elos[season][team]


def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])


def update_stats(season, team, fields):
    """
    This accepts some stats for a team and udpates the averages.

    First, we check if the team is in the dict yet. If it's not, we add it.
    Then, we try to check if the key has more than 5 values in it.
        If it does, we remove the first one
        Either way, we append the new one.
    If we can't check, then it doesn't exist, so we just add this.

    Later, we'll get the average of these items.
    """
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        if len(team_stats[season][team][key]) >= 9:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)


def get_stat(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0


def build_team_dict(folder):
    team_ids = pd.read_csv(folder + '/Teams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map


def build_season_data(all_data):
    # Calculate the elo for every game for every team, each season.
    # Store the elo per season so we can retrieve their end elo
    # later in order to predict the tournaments without having to
    # inject the prediction into this loop.
    print("Building season data.")
    for index, row in all_data.iterrows():
        # Used to skip matchups where we don't have usable stats yet.
        skip = 0

        # Get starter or previous elos.
        team_1_elo = get_elo(row['Season'], row['WTeamID'])
        team_2_elo = get_elo(row['Season'], row['LTeamID'])

        # Add 100 to the home team (# taken from Nate Silver analysis.)
        if row['WLoc'] == 'H':
            team_1_elo += 100
        elif row['WLoc'] == 'A':
            team_2_elo += 100

        # We'll create some arrays to use later.
        team_1_features = [team_1_elo]
        team_2_features = [team_2_elo]

        # Build arrays out of the stats we're tracking..
        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['WTeamID'], field)
            team_2_stat = get_stat(row['Season'], row['LTeamID'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)

        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['WFTA'] != 0 and row['LFTA'] != 0:
            stat_1_fields = {
                'score': row['WScore'],
                'fgp': row['WFGM'] / row['WFGA'] * 100,
                'fga': row['WFGA'],
                'fga3': row['WFGA3'],
                '3pp': row['WFGM3'] / row['WFGA3'] * 100,
                'ftp': row['WFTM'] / row['WFTA'] * 100,
                'or': row['WOR'],
                'dr': row['WDR'],
                'ast': row['WAst'],
                'to': row['WTO'],
                'stl': row['WStl'],
                'blk': row['WBlk'],
                'pf': row['WPF']
            }
            stat_2_fields = {
                'score': row['LScore'],
                'fgp': row['LFGM'] / row['LFGA'] * 100,
                'fga': row['LFGA'],
                'fga3': row['LFGA3'],
                '3pp': row['LFGM3'] / row['LFGA3'] * 100,
                'ftp': row['LFTM'] / row['LFTA'] * 100,
                'or': row['LOR'],
                'dr': row['LDR'],
                'ast': row['LAst'],
                'to': row['LTO'],
                'stl': row['LStl'],
                'blk': row['LBlk'],
                'pf': row['LPF']
            }
            update_stats(row['Season'], row['WTeamID'], stat_1_fields)
            update_stats(row['Season'], row['LTeamID'], stat_2_fields)

        # Now that we've added them, calc the new elo.
        new_winner_rank, new_loser_rank = calc_elo(
            row['WTeamID'], row['LTeamID'], row['Season'])
        team_elos[row['Season']][row['WTeamID']] = new_winner_rank
        team_elos[row['Season']][row['LTeamID']] = new_loser_rank

    return X, y


def find_winner(team1, team2):
    print("Looking for %d vs %d" % (team1, team2))
    # cycle through all results
    for pred in submission_data:
        parts = pred[0].split('_')
        if (int(parts[1]) == team1 and int(parts[2]) == team2) or (int(parts[2]) == team1 and int(parts[1]) == team2):
            if float(pred[1]) > .5:
              return int(parts[1]), float(pred[1])
            else:
              return int(parts[2]), (1-float(pred[1]))
    print("Could not find winner - exiting program.")
    exit(0)


def usage():
    print("Usage: %s -d <data directory> " % sys.argv[0])

def main(argv):
    folder = prediction_year = ''

    try:
        opts, args = getopt.getopt(argv, "hy:d:", ["directory=", "year="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-y':
            prediction_year = int(arg)
        elif opt in ("-d", "--directory"):
            folder = arg

    if (folder == ''):
        usage()
        sys.exit()

    if not prediction_year:
        print("Selecting current year for picks.\n")
        prediction_year = int(datetime.datetime.now().strftime("%Y"))

    team_id_map = build_team_dict(folder)

    # Now predict tournament matchups.
    print("Tournament teams for %d:" % prediction_year)
    seeds = pd.read_csv(folder + '/NCAATourneySeeds.csv')
    # for i in range(2016, 2017):
    tourney_teams = []
    tourney_seeds_map = {}
    tourney_id_to_seed_map = {}
    for index, row in seeds.iterrows():
        if row['Season'] == prediction_year:
            tourney_teams.append(row['TeamID'])
            print("Seed: %s, TeamID: %d, TeamName: %s" % (row['Seed'], int(row['TeamID']), team_id_map[row['TeamID']]))
            tourney_seeds_map[row['Seed']] = int(row['TeamID'])
            tourney_id_to_seed_map[int(row['TeamID'])] = row['Seed']

    try:
        ret = os.access(folder, os.W_OK)
        if not ret:
            sys.exit('Error with filesystem access ' + folder)
    except IOError:
        sys.exit('Error with filesystem access ' + folder)

    print("Generating results for the %d tournament." % prediction_year)

    initialize_data(prediction_year)
    season_data = pd.read_csv(folder + '/RegularSeasonDetailedResults.csv')
    tourney_data = pd.read_csv(folder + '/NCAATourneyDetailedResults.csv')
    frames = [season_data, tourney_data]
    all_data = pd.concat(frames)

    # Build the working data.
    X, y = build_season_data(all_data)

    # Fit the model.
    print("Fitting on %d samples." % len(X))

    model = linear_model.LogisticRegression()

    # Check accuracy.
    print("Doing cross-validation.")
    print(cross_val_score(
        model, numpy.array(X), numpy.array(y), cv=10, scoring='accuracy', n_jobs=-1
    ).mean())

    model.fit(X, y)


    if len(tourney_teams) == 68:
        print("Found all teams for tournament.")
    else:
        print("WARNING: Only Found %d teams for tournament. Please check input data." % len(tourney_teams))

    # Build our prediction of every matchup - useful for predicting future games.
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction = predict_winner(
                    team_1, team_2, model, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                        str(team_2)
                submission_data.append([label, prediction[0][0]])

    # Write the results.
    print("Writing %d results." % len(submission_data))
    with open(folder + '/submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data)

    # Now so that we can use this to fill out a bracket, create a readable
    # version.
    print("Outputting readable results.")
    readable = []
    less_readable = []  # A version that's easy to look up.
    for pred in submission_data:
        parts = pred[0].split('_')
        less_readable.append(
            [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
        # Order them properly.
        if pred[1] > 0.5:
            winning = int(parts[1])
            losing = int(parts[2])
            proba = pred[1]
        else:
            winning = int(parts[2])
            losing = int(parts[1])
            proba = 1 - pred[1]
        readable.append(
            [
                '%s beats %s: %f' %
                (team_id_map[winning], team_id_map[losing], proba)
            ]
        )
    with open(folder + '/readable-predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(readable)
    with open(folder + '/less-readable-predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(less_readable)

    # Run current tournament for filling out brackets
    slots = pd.read_csv(folder + '/NCAATourneySlots.csv')
    tourney_results = {}
    tourney_results_formatted = ['slot', 'team1', 'team2', 'winner', 'probability']
    for index, slot in slots.iterrows():
        if slot['Season'] == prediction_year:
            print("In Slot %s, Computing %s vs %s" % (slot['Slot'], slot['StrongSeed'], slot['WeakSeed']) )
            team1 = ''
            team2 = ''
            try:
                print("Checking winner from existing results %s vs %s" % (tourney_results[slot['StrongSeed']], tourney_results[slot['WeakSeed']]))
                team1 = tourney_results[slot['StrongSeed']]
                team2 = tourney_results[slot['WeakSeed']]
                winner, probability = find_winner(tourney_results[slot['StrongSeed']], tourney_results[slot['WeakSeed']])
            except:
                print ("Checking winner from original seeds %s vs %s" % (tourney_seeds_map[slot['StrongSeed']], tourney_seeds_map[slot['WeakSeed']]))
                team1 = tourney_seeds_map[slot['StrongSeed']]
                team2 = tourney_seeds_map[slot['WeakSeed']]
                winner, probability = find_winner(tourney_seeds_map[slot['StrongSeed']], tourney_seeds_map[slot['WeakSeed']])
            print("Winner: %d" % winner)
            tourney_seeds_map[slot['Slot']] = winner
            tourney_results[slot['Slot']] = winner
            tourney_results_formatted.append([slot['Slot'], team_id_map[team1] + "(" + tourney_id_to_seed_map[team1] + ")", team_id_map[team2] + "(" + tourney_id_to_seed_map[team2] + ")", team_id_map[winner], probability])

    with open(folder + '/tournament_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(tourney_results_formatted)


if __name__ == "__main__":
    print("====== March Madness ML Test ======")
    main(sys.argv[1:])