// Reordering for Design 2. Drag works with a mouse; the arrows work with a
// keyboard and with a screen reader. A study interface that only supports
// dragging quietly excludes participants, and the exclusion would be invisible
// in the results.
(function () {
    var list = document.getElementById('rank-list');
    var form = document.getElementById('rank-form');
    if (!list || !form) { return; }

    function renumber() {
        var rows = list.querySelectorAll('.p4-row');
        rows.forEach(function (row, i) {
            row.querySelector('.p4-rank-num').textContent = i + 1;
            row.querySelector('.up').disabled = i === 0;
            row.querySelector('.down').disabled = i === rows.length - 1;
        });
    }

    list.addEventListener('click', function (event) {
        var button = event.target.closest('.up, .down');
        if (!button) { return; }
        var row = button.closest('.p4-row');
        var sibling = button.classList.contains('up')
            ? row.previousElementSibling : row.nextElementSibling;
        if (!sibling) { return; }
        if (button.classList.contains('up')) { list.insertBefore(row, sibling); }
        else { list.insertBefore(sibling, row); }
        renumber();
        button.focus();
    });

    var dragged = null;
    list.addEventListener('dragstart', function (event) {
        dragged = event.target.closest('.p4-row');
        if (dragged) { dragged.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move'; }
    });
    list.addEventListener('dragend', function () {
        if (dragged) { dragged.classList.remove('dragging'); }
        dragged = null;
        renumber();
    });
    list.addEventListener('dragover', function (event) {
        event.preventDefault();
        if (!dragged) { return; }
        var over = event.target.closest('.p4-row');
        if (!over || over === dragged) { return; }
        var box = over.getBoundingClientRect();
        var below = event.clientY - box.top > box.height / 2;
        list.insertBefore(dragged, below ? over.nextElementSibling : over);
    });

    form.addEventListener('submit', function () {
        var fallback = document.getElementById('rank-fallback');
        if (fallback) { fallback.remove(); }
        list.querySelectorAll('.p4-row').forEach(function (row) {
            var field = document.createElement('input');
            field.type = 'hidden';
            field.name = 'order';
            field.value = row.dataset.index;
            form.appendChild(field);
        });
    });

    renumber();
})();
