
export interface CartItem {
  productId: string;
  sku: string;
  name: string;
  color: string;
  size: string;
  priceCents: number;
  imageUrl: string | null;
  quantity: number;
}

export type NewCartItem = Omit<CartItem, "quantity">;

const STORAGE_KEY = "elan_demo_cart_v1";
const CART_EVENT = "elan-cart-updated";
const MAX_QUANTITY = 99;

function isCartItem(value: unknown): value is CartItem {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const item = value as Record<string, unknown>;

  return (
    typeof item.productId === "string" &&
    typeof item.sku === "string" &&
    typeof item.name === "string" &&
    typeof item.color === "string" &&
    typeof item.size === "string" &&
    typeof item.priceCents === "number" &&
    Number.isInteger(item.priceCents) &&
    item.priceCents >= 0 &&
    (typeof item.imageUrl === "string" ||
      item.imageUrl === null) &&
    typeof item.quantity === "number" &&
    Number.isInteger(item.quantity) &&
    item.quantity >= 1 &&
    item.quantity <= MAX_QUANTITY
  );
}

export function readCart(): CartItem[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);

    if (!stored) {
      return [];
    }

    const parsed: unknown = JSON.parse(stored);

    return Array.isArray(parsed)
      ? parsed.filter(isCartItem)
      : [];
  } catch {
    return [];
  }
}

function saveCart(items: CartItem[]): void {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(items)
  );

  window.dispatchEvent(new Event(CART_EVENT));
}

export function addToCart(item: NewCartItem): void {
  const items = readCart();
  const existing = items.find(
    (current) => current.sku === item.sku
  );

  if (existing) {
    existing.quantity = Math.min(
      existing.quantity + 1,
      MAX_QUANTITY
    );
  } else {
    items.push({
      ...item,
      quantity: 1,
    });
  }

  saveCart(items);
}

export function updateCartQuantity(
  sku: string,
  quantity: number
): void {
  const items = readCart();

  if (!Number.isFinite(quantity)) {
    return;
  }

  if (quantity <= 0) {
    removeFromCart(sku);
    return;
  }

  const updated = items.map((item) =>
    item.sku === sku
      ? {
          ...item,
          quantity: Math.min(
            MAX_QUANTITY,
            Math.floor(quantity)
          ),
        }
      : item
  );

  saveCart(updated);
}

export function removeFromCart(sku: string): void {
  saveCart(
    readCart().filter((item) => item.sku !== sku)
  );
}

export function clearCart(): void {
  saveCart([]);
}

export function cartSubtotal(items: CartItem[]): number {
  return items.reduce(
    (total, item) =>
      total + item.priceCents * item.quantity,
    0
  );
}

export function subscribeToCart(
  listener: () => void
): () => void {
  window.addEventListener(CART_EVENT, listener);
  window.addEventListener("storage", listener);

  return () => {
    window.removeEventListener(CART_EVENT, listener);
    window.removeEventListener("storage", listener);
  };
}
