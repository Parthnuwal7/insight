/**
 * Display-only grouping of canonical aspects into concepts.
 *
 * `absa/aspect_canonical.py` deliberately stops at morphology (case,
 * punctuation, determiners, plurals) and refuses synonym merging, on the
 * grounds that over-merging destroys signal invisibly while fragmentation
 * at least stays visible. That reasoning is right for the stored data — so
 * this layer does not touch it. It groups only for display, and every
 * group renders its own members, so a merge this module gets wrong is
 * visible on screen rather than silently baked into the numbers.
 *
 * Two conservative rules, in order:
 *
 *   1. ABBREVIATIONS — an explicit, hand-checked map. Only cases where the
 *      short form has no other reading in a product-review context ("ui" is
 *      never anything but the interface).
 *
 *   2. GENERIC MODIFIERS — strip a leading modifier that adds no facet
 *      ("user interface" → "interface"), and only when the remainder is
 *      itself an aspect this run actually observed. "battery drain" keeps
 *      its head ("drain" is not a generic modifier of anything), and
 *      "security feature" never collapses into "feature" because
 *      "security" is not in the generic list. That restraint is the whole
 *      point: `food quality` and `sound quality` must stay distinct.
 *
 * Rule 2 is deliberately data-dependent. If a run has "user interface" but
 * no bare "interface", no group is invented — you get "user interface"
 * unchanged. Groups only ever form around a concept the run already has.
 */

/** Short forms with exactly one reading in a review corpus. */
const ABBREVIATIONS: Record<string, string> = {
  ui: "interface",
  gui: "interface",
  ux: "user experience",
  app: "app",
  application: "app",
  notif: "notification",
  notifs: "notification",
  config: "configuration",
  spec: "specification",
  docs: "documentation",
  doc: "documentation",
};

/**
 * Leading words that qualify *who* or *where*, never *which facet*.
 *
 * Adding to this list is how you make grouping more aggressive — and how
 * you break it. A word belongs here only if removing it leaves the same
 * concept for every noun it could precede. "security" fails that test
 * ("security feature" ≠ "feature"); "user" passes ("user interface" =
 * "interface").
 */
const GENERIC_MODIFIERS = new Set([
  "user",
  "overall",
  "general",
  "whole",
  "entire",
  "main",
  "basic",
  "standard",
]);

export interface AspectGroup {
  /** The concept name shown to the user. */
  label: string;
  /** Every canonical form folded into this group, label first. Rendered
   *  in the UI so a wrong merge is visible rather than silent. */
  members: string[];
}

/**
 * Build a canonical → group mapping for one run's observed aspects.
 *
 * Pass every canonical aspect the run produced (duplicates fine). The
 * returned map is total over that input: every canonical resolves to a
 * group, most of them to a single-member group that is just themselves.
 */
export function buildAspectGroups(
  canonicals: Iterable<string>,
): Map<string, AspectGroup> {
  const observed = new Set<string>();
  for (const raw of canonicals) {
    const c = raw?.trim().toLowerCase();
    if (c) observed.add(c);
  }

  // Pass 1 — abbreviations. Resolve each observed aspect to a provisional
  // label, and record which labels the run can actually anchor a group on.
  const afterAbbrev = new Map<string, string>();
  for (const c of observed) {
    afterAbbrev.set(c, ABBREVIATIONS[c] ?? c);
  }
  const anchors = new Set(afterAbbrev.values());

  // Pass 2 — generic modifiers, only onto an anchor the run already has.
  const labelOf = new Map<string, string>();
  for (const c of observed) {
    let label = afterAbbrev.get(c)!;
    const tokens = label.split(/\s+/);
    while (
      tokens.length > 1 &&
      GENERIC_MODIFIERS.has(tokens[0]) &&
      anchors.has(tokens.slice(1).join(" "))
    ) {
      tokens.shift();
      label = tokens.join(" ");
    }
    labelOf.set(c, label);
  }

  // Invert into groups, keeping the label itself first among members.
  const membersOf = new Map<string, string[]>();
  for (const [canonical, label] of labelOf) {
    if (!membersOf.has(label)) membersOf.set(label, []);
    membersOf.get(label)!.push(canonical);
  }
  for (const members of membersOf.values()) {
    members.sort((a, b) => a.localeCompare(b));
  }

  const groups = new Map<string, AspectGroup>();
  for (const [canonical, label] of labelOf) {
    const members = membersOf.get(label)!;
    groups.set(canonical, {
      label,
      members: [label, ...members.filter((m) => m !== label)],
    });
  }
  return groups;
}

/** Aspects that name the product rather than a facet of it. Counting them
 *  is not wrong, but they can never be acted on — "fix the app" is not a
 *  finding — so the UI marks them instead of ranking them as signal. */
const CONTAINER_ASPECTS = new Set(["app", "product", "service", "software", "thing"]);

export function isContainerAspect(label: string): boolean {
  return CONTAINER_ASPECTS.has(label.trim().toLowerCase());
}
