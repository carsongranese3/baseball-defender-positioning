import os
import subprocess
import sys

def clear_screen():
    # Clears the terminal for a clean UI experience
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 50)
    print("      BASEBALL DEFENSIVE STRATEGY ENGINE v2.0")
    print("=" * 50)

def main_menu():
    while True:
        clear_screen()
        print_header()
        print("\n [1] Add / Update Batter Data")
        print(" [2] Add / Update Pitcher Data")
        print(" [3] Run Defensive Simulator (Matchup Mode)")
        print(" [4] Exit")
        
        choice = input("\nSelect an option: ").strip()

        if choice == '1':
            print("\n--- Launching Batter Downloader ---")
            # Replace 'get_batter_data.py' with your actual filename
            subprocess.run([sys.executable, "getBatterData.py"])
            input("\nPress Enter to return to menu...")

        elif choice == '2':
            print("\n--- Launching Pitcher Downloader ---")
            # Replace 'get_pitcher_data.py' with your actual filename
            subprocess.run([sys.executable, "getPitcherData.py"])
            input("\nPress Enter to return to menu...")

        elif choice == '3':
            print("\n--- Launching Simulator ---")
            # Runs your display.py script
            subprocess.run([sys.executable, "display.py"])
            input("\nPress Enter to return to menu...")

        elif choice == '4':
            print("\nShutting down. Good luck on the field!")
            break
        
        else:
            print("\n[!] Invalid choice. Please pick 1-4.")
            input("Press Enter to try again...")

if __name__ == "__main__":
    main_menu()