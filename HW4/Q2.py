import random
import numpy as np

def deal(deck): # Deal two cards from the current deck
    hand = []
    for i in range(2):
        random.shuffle(deck)
        card = deck.pop()
        if card == 11:card = "J"
        if card == 12:card = "Q"
        if card == 13:card = "K"
        if card == 14:card = "A"
        hand.append(card)       
    return hand

def total(hand): # Compute the total sum of the current hand
    total = 0
    for card in hand:
        if card in ["J", "Q", "K"]:
            total += 10
        elif card == "A":
            total += 11
        else:
            total += card
    return total

def hit(deck): # Hit another card from the current deck
    card = deck.pop()
    if card == 11:
        card = "J"
    elif card == 12:
        card = "Q"
    elif card == 13:
        card = "K"
    elif card == 14:
        card = "A"
    return card

def reward(player_hand, dealer_hand): # Compute the reward - 1 if won, 0 otherwise
    player = total(player_hand)
    dealer = total(dealer_hand)
    if player == 21: # If the gambler got a blackjack, the gambler won
        return 1
    if player < 21 and dealer > 21: # If dealer burnt out, the gambler won
        return 1
    if player < 21 and dealer < 21 and dealer < player: # If neither burnt out, but gambler has higher sum, the gambler won
        return 1
    return 0

def game(p, Q, t):
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*4 # Init. the deck
    player_hand = deal(deck) # Deal the gambler
    dealer_hand = deal(deck) # Deal the dealer
    action, Q = p(player_hand, Q, t) # Choose an action and update the Q-function
    while action == 1: # As long as policy wants to hit, continue
        prev_hand = player_hand.copy() # Previous hand required for SARSA
        player_hand.append(hit(deck)) # Hit the gambler
        action, Q = p(player_hand, Q, t, prev_hand, 1) # Choose an action and update the Q-function
    if total(dealer_hand) < 16: # Naive policy of the dealer - keep hitting until hitting a sum greater than 16
        dealer_hand.append(hit(deck))
    r = reward(player_hand, dealer_hand) # Compute reward
    return r, player_hand, Q # Return reward, last hand and updated Q-function

def main():
    random.seed(42)
    l = len(range(4, 33))
    Q = [[0 for i in range(2)] for j in range(l)] # Accounting for states available after hits from 21
    N = 100000
    lr = 0.01
    gamma = 1

    def p(player_hand, Q, t, prev_hand=None, prev_action=None): # Choose an action and update the Q-function
        idx = total(player_hand)-4 # Index used to access the Q-function - s_{t+1} of the SARSA algorithm
        eps = (1 / t**0.1) / 2 # Decaying epsilon
        action = np.argmax(Q[idx]) # Greey action
        if random.random() < eps: # If the non-greey action is to be chosen
            if action == 1: # The variable action is a_{t+1} of the SARSA algorithm
                action = 0
            else:
                action = 1
        if idx > 17: # This implies the sum is greater than 21 -> no more hits
            action = 0
        if prev_hand: # If the current hand isn't the first one, update Q-function - s_{t} of the SARSA algorithm
            # Q_{t+1}(s_{t},a_{t}) = Q_{t}(s_{t},a_{t}) + lr * (r_t + Q_{t}(s_{t+1}, a_{t+1}) - Q_{t}(s_{t},a_{t}))
            # prev_action is a_{t} of the SARSA algorithm and won't be None
            # 0 is r_{t} since the gambler hasn't won yet
            prev_idx = total(prev_hand)-4 # s_{t} of the SARSA algorithm
            Q[prev_idx][prev_action] = Q[prev_idx][prev_action] + lr * (0 + gamma*Q[idx][action] - Q[prev_idx][prev_action])
        return action, Q

    for t in range(N):
        r, player_hand, Q = game(p, Q, t+1) # Run an episode, keep term. reward, final state and updated Q-function
        idx = total(player_hand) - 4 # s_{t} of the SARSA algorithm
        # The final action a_{t} is always 0
        # The final reward r_{t} is r
        # The final state has Q-value of 0 
        Q[idx][0] = Q[idx][0] + lr * (r + gamma * 0 - Q[idx][0])
    
    for j in range(l):
        print(f'score: {range(4, 33)[j]}, action: {0}, prob: {Q[j][0]}, action: {1}, prob: {Q[j][1]}')

if __name__ == "__main__":
    main()