import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import random

class ChordProgressionAI:
    def __init__(self):
        self.genreChordProgressions = {
            "rock": [["E", "A", "D", "G"], ["C", "G", "Am", "F"], ["D", "A", "G", "C"]],
            "pop": [["C", "G", "Am", "F"], ["G", "D", "Em", "C"], ["A", "E", "F#m", "D"]],
            "jazz": [["Dm7", "G7", "Cmaj7", "A7"], ["Cmaj7", "B7", "E7", "A7"], ["Fmaj7", "Bb7", "Ebmaj7", "Ab7"]],
            "blues": [["E7", "A7", "B7"], ["A7", "D7", "E7"], ["C7", "F7", "G7"]],
            "classical": [["C", "G", "Am", "F"], ["F", "C", "G", "Am"], ["G", "D", "Em", "C"]],
            "metal": [["E5", "G5", "A5", "D5"], ["D5", "C5", "G5", "A5"], ["F5", "Bb5", "C5", "G5"]],
            "country": [["G", "C", "D", "Em"], ["D", "A", "G", "C"], ["C", "G", "D", "A"]],
            "reggae": [["A", "D", "E", "G"], ["C", "F", "G", "Am"], ["D", "A", "B", "E"]],
            "hip-hop": [["C", "F", "G", "Am"], ["D", "A", "Bm", "G"], ["E", "C#m", "A", "B"]]
        }
        self.model = self.buildModel()
        self.trainModel()  

    def buildModel(self):
        model = nn.Sequential(
            nn.Linear(9, 50),  
            nn.ReLU(),          
            nn.Linear(50, 9)    
        )
        return model

    def trainModel(self):
        trainingData = [
            ("rock", [1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("pop", [0, 1, 0, 0, 0, 0, 0, 0, 0]),
            ("jazz", [0, 0, 1, 0, 0, 0, 0, 0, 0]),
            ("blues", [0, 0, 0, 1, 0, 0, 0, 0, 0]),
            ("classical", [0, 0, 0, 0, 1 , 0, 0, 0, 0]),
            ("metal", [0, 0, 0, 0, 0, 1, 0, 0, 0]),
            ("country", [0, 0, 0, 0, 0, 0, 1, 0, 0]),
            ("reggae", [0, 0, 0, 0, 0, 0, 0, 1, 0]),
            ("hip-hop", [0, 0, 0, 0, 0, 0, 0, 0, 1])
        ]
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(100): 
            for genre, target in trainingData:
                inputTensor = torch.tensor(target, dtype=torch.float32)
                targetTensor = torch.tensor([1.0 if g == genre else 0.0 for g in self.genreChordProgressions.keys()][:9], dtype=torch.float32)
                optimizer.zero_grad()
                output = self.model(inputTensor)
                loss = criterion(output, targetTensor)
                loss.backward()
                optimizer.step()

    def suggestChordProgression(self, genre):
        progressions = self.genreChordProgressions.get(genre.lower(), [["C", "G", "Am", "F"]])
        return random.choice(progressions) 

class ChordApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sugestão de Progressão de Acordes")
        self.root.geometry("400x300")  
        self.root.configure(bg="#f0f0f0")  
        self.musicAI = ChordProgressionAI()

        self.label = tk.Label(root, text="Digite o gênero musical:", bg="#f0f0f0", font=("Arial", 14))
        self.label.pack(pady=10)
        self.genreEntry = tk.Entry(root, font=("Arial", 12), width=30)
        self.genreEntry.pack(pady=5)
        self.suggestButton = tk.Button(root, text="Sugerir Progressão", command=self.suggestProgression, bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.suggestButton.pack(pady=10)
        self.resultLabel = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 12))
        self.resultLabel.pack(pady=10)

    def suggestProgression(self):
        genre = self.genreEntry.get()
        if genre:
            suggestedProgression = self.musicAI.suggestChordProgression(genre)
            self.resultLabel.config(text="Sugestão de progressão de acordes: " + " - ".join(suggestedProgression))
        else:
            messagebox.showwarning("Entrada inválida", "Por favor, insira um gênero musical.")

def main():
    root = tk.Tk()
    app = ChordApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
