/* Lute functions. */

/**
 * The current term index (either clicked or hovered over)
 */
let LUTE_CURR_TERM_DATA_ORDER = -1;  // initially not set.

/**
 * Lute has 2 different "modes" when reading:
 * - LUTE_HOVERING = true: Hover mode, not selecting
 * - LUTE_HOVERING = false: Word clicked, or click-drag
 */ 
let LUTE_HOVERING = true;

/**
 * When the reading pane is first loaded, it is set in "hover mode",
 * meaning that when the user hovers over a word, that word becomes
 * the "active word" -- i.e., status update keyboard shortcuts should
 * operate on that hovered word, and as the user moves the mouse
 * around, the "active word" changes.  When a word is clicked, though,
 * there can't be any "hover changes", because the user should be
 * editing the word in the Term edit pane, and has to consciously
 * disable the "clicked word" mode by hitting ESC or RETURN.
 *
 * The full page text is often reloaded via ajax, e.g. when the user
 * saves an edited term, or the status is updated with a hotkey.
 * The template lute/templates/read/page_content.html calls
 * this method on reload to reset the cursor etc.
 */
function start_hover_mode(should_clear_frames = true) {
  // console.log('CALLING RESET');
  LUTE_HOVERING = true;

  $('span.kwordmarked').removeClass('kwordmarked');

  const curr_word = $('span.word').filter(function() {
    return _get_order($(this)) == LUTE_CURR_TERM_DATA_ORDER;
  });
  if (curr_word.length == 1) {
    const w = $(curr_word[0]);
    $(w).addClass('wordhover');
    apply_status_class($(w));
  }

  if (should_clear_frames) {
    $('#wordframeid').attr('src', '/read/empty');
    $('#dictframeid').attr('src', '/read/empty');
  }

  clear_newmultiterm_elements();

  // https://stackoverflow.com/questions/35022716/keydown-not-detected-until-window-is-clicked
  $(window).focus();
}


/**
 * On new page load, the cursor should be reset to the start of the page.
 */
function reset_cursor() {
  LUTE_CURR_TERM_DATA_ORDER = -1;
  $('span.wordhover').removeClass('wordhover');
  $('span.kwordmarked').removeClass('kwordmarked');
}


/** 
 * Prepare the interaction events with the text.
 *
 * pos = position hash, e.g.
 * {my: 'center bottom', at: 'center top-10', collision: 'flipfit flip'}
 */
function prepareTextInteractions(pos) {
  const t = $('#thetext');
  // Using "t.on" here because .word elements
  // are added and removed dynamically, and "t.on"
  // ensures that events remain for each element.
  t.on('mousedown', '.word', select_started);
  t.on('mouseover', '.word', select_over);
  t.on('mouseup', '.word', select_ended);

  t.on('mouseover', '.word', hover_over);
  t.on('mouseout', '.word', hover_out);

  if (!_show_highlights()) {
    t.on('mouseover', '.word', hover_over_add_status_class);
  }

  $(document).on('keydown', handle_keydown);

  $('#thetext').tooltip({
    position: pos,
    items: '.word.showtooltip',
    show: { easing: 'easeOutCirc' },
    content: function (setContent) { tooltip_textitem_hover_content($(this), setContent); }
  });
}


/**
 * Build the html content for jquery-ui tooltip.
 */
let tooltip_textitem_hover_content = function (el, setContent) {
  elid = parseInt(el.data('wid'));
  $.ajax({
    url: `/read/termpopup/${elid}`,
    type: 'get',
    success: function(response) {
      setContent(response);
    }
  });
}


function showEditFrame(el, extra_args = {}) {
  const lid = parseInt(el.data('lang-id'));

  let text = extra_args.textparts ?? [ el.data('text') ];
  const sendtext = text.join('');

  let extras = Object.entries(extra_args).
      map((p) => `${p[0]}=${encodeURIComponent(p[1])}`).
      join('&');

  const url = `/read/termform/${lid}/${sendtext}?${extras}`;
  // console.log('go to url = ' + url);

  top.frames.wordframe.location.href = url;
}


let _get_order = function(el) { return parseInt(el.data('order')); };

let save_curr_data_order = function(el) {
  LUTE_CURR_TERM_DATA_ORDER = _get_order(el);
}


/* ========================================= */
/** Status highlights.
 * There is a "show_highlights" UserSetting.
 *
 * If showing highlights, then the highlights are shown when the page
 * is rendered; otherwise, they're only shown/removed on hover enter/exit.
 *
 * User setting "show_highlights" is rendered in read/index.html. */

/** True if show_highlights setting is True. */
let _show_highlights = function() {
  return ($("#show_highlights").text().toLowerCase() == "true");
}

/**
 * Terms have data-status-class attribute.  If highlights should be shown,
 * then add that value to the actual span. */
function add_status_classes() {
  if (!_show_highlights())
    return;
  $('span.word').toArray().forEach(function (w) {
    apply_status_class($(w));
  });
}

/** Add the data-status-class to the term's classes. */
let apply_status_class = function(el) {
  el.addClass(el.data("status-class"));
}

/** Remove the status from elements, if not showing highlights. */
let remove_status_highlights = function() {
  if (_show_highlights()) {
    /* Not removing anything, always showing highlights. */
    return;
  }
  $('span.word').toArray().forEach(function (m) {
    el = $(m);
    el.removeClass(el.data("status-class"));
  });
}

/** Hover and term status classes */
function hover_over_add_status_class(e) {
  remove_status_highlights();
  apply_status_class($(this));
}


/* ========================================= */
/** Hovering */

function hover_over(e) {
  if (! LUTE_HOVERING)
    return;
  $('span.wordhover').removeClass('wordhover');
  $(this).addClass('wordhover');
  save_curr_data_order($(this));
}

function hover_out(e) {
  if (! LUTE_HOVERING)
    return;
  $('span.wordhover').removeClass('wordhover');
}


/* ========================================= */
/** Clicking */

let word_clicked = function(el, e) {
  el.removeClass('wordhover');
  save_curr_data_order(el);
  if (el.hasClass('kwordmarked')) {
    el.removeClass('kwordmarked');
    const nothing_marked = $('span.kwordmarked').length == 0;
    if (nothing_marked) {
      el.addClass('wordhover');
      start_hover_mode();
    }
  }
  else {
    if (! e.shiftKey) {
      $('span.kwordmarked').removeClass('kwordmarked');
      showEditFrame(el);
    }
    el.addClass('kwordmarked');
    el.removeClass('hasflash');
  }
}


/* ========================================= */
/** Multiword selection */

let selection_start_el = null;

let clear_newmultiterm_elements = function() {
  $('.newmultiterm').removeClass('newmultiterm');
  selection_start_el = null;
}

function select_started(e) {
  LUTE_HOVERING = false;
  $('span.wordhover').removeClass('wordhover');
  clear_newmultiterm_elements();
  $(this).addClass('newmultiterm');
  selection_start_el = $(this);
  save_curr_data_order($(this));
}

let get_selected_in_range = function(start_el, end_el) {
  const [startord, endord] = [_get_order(start_el), _get_order(end_el)].sort();
  const selected = $('span.textitem').filter(function() {
    const ord = _get_order($(this));
    return ord >= startord && ord <= endord;
  });
  return selected;
};

function select_over(e) {
  if (selection_start_el == null)
    return;  // Not selecting
  $('.newmultiterm').removeClass('newmultiterm');
  const selected = get_selected_in_range(selection_start_el, $(this));
  selected.addClass('newmultiterm');
}

function select_ended(e) {
  // Handle single word click.
  if (selection_start_el.attr('id') == $(this).attr('id')) {
    clear_newmultiterm_elements();
    word_clicked($(this), e);
    return;
  }

  $('span.kwordmarked').removeClass('kwordmarked');

  const selected = get_selected_in_range(selection_start_el, $(this));
  if (e.shiftKey) {
    copy_text_to_clipboard(selected.toArray());
    start_hover_mode(false);
    return;
  }

  const textparts = selected.toArray().map((el) => $(el).text());
  const text = textparts.join('').trim();
  if (text.length > 250) {
    alert(`Selections can be max length 250 chars ("${text}" is ${text.length} chars)`);
    start_hover_mode();
    return;
  }

  showEditFrame(selection_start_el, { textparts: textparts });
  selection_start_el = null;
}


/********************************************/
// Keyboard navigation.

/** Get the rest of the textitems in the current active/hovered word's
 * sentence or paragraph, or null if no selection. */
let get_textitems_spans = function(e) {
  let elements = $('span.kwordmarked, span.newmultiterm, span.wordhover');
  elements.sort((a, b) => _get_order($(a)) - _get_order($(b)));
  if (elements.length == 0)
    return null;

  const w = elements[0];
  const attr_name = e.shiftKey ? 'paragraph-id' : 'sentence-id';
  const attr_value = $(w).data(attr_name);
  return $(`span.textitem[data-${attr_name}="${attr_value}"]`).toArray();
};

/** Copy the text of the textitemspans to the clipboard, and add a
 * color flash. */
let handle_copy = function(e) {
  tis = get_textitems_spans(e);
  if (tis != null)
    copy_text_to_clipboard(tis);
}

let copy_text_to_clipboard = function(textitemspans) {
  const copytext = textitemspans.map(s => $(s).text()).join('');

  // console.log('copying ' + copytext);
  var textArea = document.createElement("textarea");
  textArea.value = copytext;
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("Copy");
  textArea.remove();

  const removeFlash = function() {
    // console.log('removing flash');
    $('span.flashtextcopy').addClass('wascopied'); // for acceptance testing.
    $('span.flashtextcopy').removeClass('flashtextcopy');
  };

  // Add flash, set timer to remove.
  removeFlash();
  textitemspans.forEach(function (t) {
    $(t).addClass('flashtextcopy');
  });
  setTimeout(() => removeFlash(), 1000);

  $('#wordframeid').attr('src', '/read/flashcopied');
}


let move_cursor = function(shiftby) {
  // Use the first clicked element, or the hovered if none clicked.
  let elements = $('span.kwordmarked, span.newmultiterm');
  let hovered_els = $('span.wordhover');
  if (elements.length == 0 && hovered_els.length > 0)
    elements = hovered_els;
  elements.sort((a, b) => _get_order($(a)) - _get_order($(b)));
  const curr = (elements.length == 0) ? null : elements[0];

  let words = $('span.word');
  words.sort((a, b) => _get_order($(a)) - _get_order($(b)));

  let _get_new_index = function(curr) {
    if (curr == null)
      return 0;
    const pid = $(curr).attr('id');
    let i = words.toArray().findIndex(e => $(e).attr('id') == pid) + shiftby;
    i = Math.max(i, 0); // ensure >= 0
    i = Math.min(i, words.length - 1);  // within array.
    return i;
  }
  const target = $(words[_get_new_index(curr)]);

  // Adjust all screen state.
  $('span.newmultiterm').removeClass('newmultiterm');
  $('span.kwordmarked').removeClass('kwordmarked');
  remove_status_highlights();
  target.addClass('kwordmarked');
  save_curr_data_order(target);
  apply_status_class(target);
  $(window).scrollTo(target, { axis: 'y', offset: -150 });
  showEditFrame(target, { autofocus: false });
}


let show_translation = function(e) {
  tis = get_textitems_spans(e);
  if (tis == null)
    return;
  const sentence = tis.map(s => $(s).text()).join('');

  const userdict = $('#translateURL').text();
  if (userdict == null || userdict == '')
    console.log('No userdict for lookup.  ???');

  // console.log(userdict);
  const url = userdict.replace('###', encodeURIComponent(sentence));
  if (url[0] == '*') {
    const finalurl = url.substring(1);  // drop first char.
    const settings = 'width=800, height=400, scrollbars=yes, menubar=no, resizable=yes, status=no';
    window.open(finalurl, 'dictwin', settings);
  }
  else {
    top.frames.dictframe.location.href = url;
  }
}


/* Change to the next theme, and reload the page. */
function next_theme() {
  $.ajax({
    url: '/theme/next',
    type: 'post',
    dataType: 'JSON',
    contentType: 'application/json',
    success: function(response) {
      location.reload();
    },
    error: function(response, status, err) {
      const msg = {
        response: response,
        status: status,
        error: err
      };
      console.log(`failed: ${JSON.stringify(msg, null, 2)}`);
    }
  });

}


/* Toggle highlighting, and reload the page. */
function toggle_highlight() {
  $.ajax({
    url: '/theme/toggle_highlight',
    type: 'post',
    dataType: 'JSON',
    contentType: 'application/json',
    success: function(response) {
      location.reload();
    },
    error: function(response, status, err) {
      const msg = {
        response: response,
        status: status,
        error: err
      };
      console.log(`failed: ${JSON.stringify(msg, null, 2)}`);
    }
  });
}


function handle_keydown (e) {
  if ($('span.word').length == 0) {
    // console.log('no words, exiting');
    return; // Nothing to do.
  }

  // Map of key codes (e.which) to lambdas:
  let map = {};

  const kESC = 27;
  const kRETURN = 13;
  const kLEFT = 37;
  const kRIGHT = 39;
  const kUP = 38;
  const kDOWN = 40;
  const kC = 67; // C)opy
  const kT = 84; // T)ranslate
  const kM = 77; // The(M)e
  const kH = 72; // Toggle H)ighlight
  const k1 = 49;
  const k2 = 50;
  const k3 = 51;
  const k4 = 52;
  const k5 = 53;
  const kI = 73;
  const kW = 87;

  map[kESC] = () => start_hover_mode();
  map[kRETURN] = () => start_hover_mode();
  map[kLEFT] = () => move_cursor(-1);
  map[kRIGHT] = () => move_cursor(+1);
  map[kUP] = () => increment_status_for_selected_elements(e, +1);
  map[kDOWN] = () => increment_status_for_selected_elements(e, -1);
  map[kC] = () => handle_copy(e);
  map[kT] = () => show_translation(e);
  map[kM] = () => next_theme();
  map[kH] = () => toggle_highlight();
  map[k1] = () => update_status_for_marked_elements(1);
  map[k2] = () => update_status_for_marked_elements(2);
  map[k3] = () => update_status_for_marked_elements(3);
  map[k4] = () => update_status_for_marked_elements(4);
  map[k5] = () => update_status_for_marked_elements(5);
  map[kI] = () => update_status_for_marked_elements(98);
  map[kW] = () => update_status_for_marked_elements(99);

  if (e.which in map) {
    let a = map[e.which];
    a();
  }
  else {
    // console.log('unhandled key ' + e.which);
  }
}


/**
 * If the term editing form is visible when reading, and a hotkey is hit,
 * the form status should also update.
 */
function update_term_form(el, new_status) {
  const sel = 'input[name="status"][value="' + new_status + '"]';
  var radioButton = top.frames.wordframe.document.querySelector(sel);
  if (radioButton) {
    radioButton.click();
  }
  else {
    // Not found - user might just be hovering over the element,
    // or multiple elements selected.
    // console.log("Radio button with value " + new_status + " not found.");
  }
}


function update_status_for_marked_elements(new_status) {
  let elements = $('span.kwordmarked').toArray().concat($('span.wordhover').toArray());
  let updates = [ make_status_update_hash(new_status, elements) ]
  post_bulk_update(updates);
}


function make_status_update_hash(new_status, elements) {
  const texts = elements.map(el => $(el).text());
  return {
    new_status: new_status,
    terms: texts
  }
}


function post_bulk_update(updates) {
  if (updates.length == 0) {
    // console.log("No updates.");
    return;
  }
  let elements = $('span.kwordmarked').toArray().concat($('span.wordhover').toArray());
  if (elements.length == 0)
    return;
  const firstel = $(elements[0]);
  const first_status = updates[0].new_status;
  const langid = firstel.data('lang-id');
  const selected_ids = $('span.kwordmarked').toArray().map(el => $(el).attr('id'));

  data = JSON.stringify({
    langid: langid, updates: updates
  });

  let re_mark_selected_ids = function() {
    for (let i = 0; i < selected_ids.length; i++) {
      let el = $(`#${selected_ids[i]}`);
      el.addClass('kwordmarked');
    }
    if (selected_ids.length > 0)
      $('span.wordhover').removeClass('wordhover');
  };

  let reload_text_div = function() {
    const bookid = $('#book_id').val();
    const pagenum = $('#page_num').val();
    const url = `/read/renderpage/${bookid}/${pagenum}`;
    const repel = $('#thetext');
    repel.load(url, re_mark_selected_ids);
  };

  $.ajax({
    url: '/term/bulk_update_status',
    type: 'post',
    data: data,
    dataType: 'JSON',
    contentType: 'application/json',
    success: function(response) {
      reload_text_div();
      if (elements.length == 1) {
        update_term_form(firstel, first_status);
      }
    },
    error: function(response, status, err) {
      const msg = {
        response: response,
        status: status,
        error: err
      };
      console.log(`failed: ${JSON.stringify(msg, null, 2)}`);
    }
  });

}


/**
 * Change status using arrow keys for *select* (clicked) elements only,
 * *not* hovered elements.
 *
 * When hovering, clicking an arrow should just scroll the screen, because
 * the user is *kind of passively* viewing content.
 * If the user has clicked on an element (or used arrow keys), they're actively
 * focused on it.
 */
function increment_status_for_selected_elements(e, shiftBy) {
  const elements = Array.from(document.querySelectorAll('span.kwordmarked'));
  if (elements.length == 0)
    return;

  // Don't scroll screen.  If screen scrolling happens, then pressing
  // "up" will both scroll up *and* change the status the selected term,
  // which is odd.
  e.preventDefault();

  const validStatuses = ['status0', 'status1', 'status2', 'status3', 'status4', 'status5', 'status99'];

  // Build payloads to update for each unique status that will be changing
  let payloads = {};

  elements.forEach((element) => {
    let statusClass = element.dataset.statusClass;
    
    if (!statusClass || !validStatuses.includes(statusClass)) return;

    payloads[statusClass] ||= [];
    payloads[statusClass].push(element);
  })

  // Convert payloads to update hashes.
  let updates = []

  Object.keys(payloads).forEach((key) => {
    let originalIndex = validStatuses.indexOf(key);

    if (originalIndex == -1) return;
    
    newIndex = Math.max(0, Math.min((validStatuses.length-1), originalIndex+shiftBy));

    if (newIndex != originalIndex) {
      const newStatusCode = Number(validStatuses[newIndex].replace(/\D/g, ''));

      // Can't set status to 0, that implies term deletion, which is a different issue
      // (at the moment).
      // TODO delete term from reading screen: setting to 0 could equal deleting term.
      if (newStatusCode != 0) {
        updates.push(make_status_update_hash(newStatusCode, payloads[key]));
      }
    }
  });

  post_bulk_update(updates);
}
