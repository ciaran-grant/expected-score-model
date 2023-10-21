

# Load Library
library(AFL)
library(data.table)

# Load AFL Stats Object
Stats = load_stats()

# Match Chain - event data, x, y locations
##### Chain Data


# Collect all Match Chains
round_metadata_list = c(".consolidated_metadata",".__enclos_env__","Metadata","clone","initialize")

match_chains = list()
for (season in c("2021", "2022")){
  print(season)
  round_list = sort(names(Stats)[grepl(paste0("^",season), names(Stats))])
  for (round in round_list){
    print(round)
    Stats_round = Stats[[round]]
    match_list= sort(setdiff(names(Stats_round), round_metadata_list))
    for (match in match_list){
      Stats_match = Stats_round[[match]]
      print(match)
      # print("AFL API Match Stats")
      match_chains = rbind.fill(match_chains, Stats_match[['AFL_API_Match_Chains']])
    }
  }
}
match_chains = data.table(match_chains)

## Types
match_chains[, .N, Description]

## Shots
shots = match_chains[Shot_At_Goal == TRUE]
shots[, .N, Final_State]

## Export
write.csv(match_chains, "/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/data/match_chains.csv", row.names = FALSE)
write.csv(shots, "/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/data/shot_chains.csv", row.names = FALSE)

