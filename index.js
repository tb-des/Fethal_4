// Sélectionner toutes les cases
const cases = document.querySelectorAll('.grid-item');
var $is_turn = 0;
const $player_color = 'yellow';
const $ai_color = 'red';

// Trouver la case disponible la plus basse dans la colonne
function findLowestAvailableCase(caseIndex) {
    let lowestCase = null;
    for (let i = cases.length + caseIndex%7 - 7; i >= 0; i -= 7) {
        const caseElement = cases[i];
        if (caseElement.style.backgroundColor !== $player_color && caseElement.style.backgroundColor !== $ai_color) {
            lowestCase = caseElement;
            break;
        }
    }
    return lowestCase;
}

function tablecase() {
    let table = [];
    for (let i=0; i<cases.length; i++) {
        if (cases[i].style.backgroundColor == $player_color) {
            table.push(1);
        }
        else if (cases[i].style.backgroundColor == $ai_color) {
            table.push(-1);
        }
        else {
            table.push(0);
        }
    }
}

// Parcourir toutes les cases
cases.forEach(caseElement => {
    // Ajouter un écouteur d'événement pour le clic
    caseElement.addEventListener('click', () => {
        // Vérifier si la case est déjà jaune
        if ($is_turn == 0) {
            if (caseElement.style.backgroundColor !== $player_color || caseElement.style.backgroundColor !== $ai_color) {
                findLowestAvailableCase(Array.from(cases).indexOf(caseElement)).style.backgroundColor = $player_color;
                $is_turn = 1;
            }
        }
        else if ($is_turn == 1) {
            
        }
    });
});

