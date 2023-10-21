

# Load Library
library(AFL)
library(data.table)
library(plyr)

# Check if latest data pulled
new_data <- AFL::check_dropbox_data()

# Get latest round data
AFL::pull_dropbox_data()

# Load AFL Stats Object
Stats = load_stats()

# Collect all Match Chains
round_metadata_list = c(".consolidated_metadata",".__enclos_env__","Metadata","clone","initialize")

match_chains = list()
for (season in c("2021", "2022", "2023")){
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
## Export
write.csv(match_chains, "/Users/ciaran/Documents/Projects/AFL/data/match_chains.csv", row.names = FALSE)

# Collect specific round Match Chains
# round_ID <- c("202320")
# round_chains = list()
# for (round in round_ID){
#   print(round)
#   Stats_round = Stats[[round]]
#   match_list= sort(setdiff(names(Stats_round), round_metadata_list))
#   for (match in match_list){
#     Stats_match = Stats_round[[match]]
#     print(match)
#     # print("AFL API Match Stats")
#     round_chains = rbind.fill(round_chains, Stats_match[['AFL_API_Match_Chains']])
#   }
#   round_chains = data.table(round_chains)
#   write.csv(round_chains, paste0("/Users/ciaran/Documents/Projects/AFL/git-repositories/afl-player-ratings/data/round_chains_", round, ".csv"), row.names = FALSE)
# }



