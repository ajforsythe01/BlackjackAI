import os
import random
from Blackjack import Blackjack

def generate_data(num_games):
    actions = []
    player_totals = []
    dealer_upcards = []
    soft = []

    for game_number in range(1, num_games + 1):
        print(f"Game: {game_number}")
        bj = Blackjack()

        # Player's turn
        if bj.state == bj.State.PLAYING:
            while bj.state == bj.State.PLAYING:
                player_totals.append(bj.player.total())
                dealer_upcards.append(bj.dealer.value(bj.dealer.hand[0]))
                if bj.player.is_soft():
                    soft.append(1)
                else:
                    soft.append(0)

                if bj.player.check21() == True:
                    action = 'h'
                else:
                    action = random.choice(['h', 's'])
                    
                    if action == 'h':
                        bj.hit()
                        if bj.player.total() > 21:
                            action = 's'
                    else:
                        bj.stand()
                        if bj.state == bj.State.LOSE:
                            action = 'h'         
            
                actions.append(action)
        else:
            player_totals.append(bj.player.total())
            dealer_upcards.append(bj.dealer.value(bj.dealer.hand[0]))
            if bj.player.is_soft():
                soft.append(1)
            else:
                soft.append(0)

            actions.append('s')

    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Write logs to files
    with open("data\\actions_log.txt", "w") as actions_file:
        for data in actions:
            actions_file.write(data +'\n')

    with open("data\\totals.txt", "w") as totals_file:
        for i,data in enumerate(player_totals):
            totals_file.write(f"[{data},{dealer_upcards[i]},{soft[i]}]\n")

if __name__ == "__main__":
    generate_data(100000)
